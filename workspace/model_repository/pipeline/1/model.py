"""
dots.mocr — Triton Python backend (pure PyTorch / HuggingFace) with dynamic batching

When Triton's dynamic_batching accumulates multiple pending requests it calls
execute() with all of them at once.  We run them as a single model.generate()
call so the GPU processes the whole batch in parallel — much better utilisation
than one request at a time.

External interface (unchanged):
  Inputs : PROMPT (string [1]), IMAGE_B64 (string [1]), REQUEST_ID (string [1], optional)
  Output : TEXT (string [1])

Cancellation: checked every 5 generation steps; if ALL requests in the batch
are cancelled the generation stops early.  Individual cancellation within a
running batch is not possible with HuggingFace generate().
"""

import base64
import io
import json
import os
import re
import threading

import numpy as np
import triton_python_backend_utils as pb_utils


def _to_str(x) -> str:
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


class TritonPythonModel:
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, args: dict) -> None:
        model_config = json.loads(args.get("model_config", "{}"))
        params = model_config.get("parameters", {})

        def _p(key: str, default: str) -> str:
            return params.get(key, {}).get("string_value", default)

        self.model_name     = _p("model_name", "rednote-hilab/dots.mocr")
        self.max_new_tokens = int(_p("max_tokens", "24000"))

        device_id   = str(args.get("model_instance_device_id", "0"))
        self.device = f"cuda:{device_id}"

        log = pb_utils.Logger
        log.log_info(f"[pipeline] Loading {self.model_name!r} on {self.device}")

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        # Left-pad so batched inputs of different lengths align on the right
        self.processor.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        # Compile for better decode throughput (~20-30% speedup on repetitive shapes)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            log.log_info("[pipeline] torch.compile applied")
        except Exception as exc:
            log.log_warn(f"[pipeline] torch.compile skipped: {exc}")

        # Log free memory so operators can tune max_batch_size
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(self.device)
            log.log_info(
                f"[pipeline] GPU memory after model load — "
                f"free: {free/1e9:.1f} GB / total: {total/1e9:.1f} GB"
            )

        log.log_info(f"[pipeline] Model ready on {self.device}")

        # Redis — optional; graceful degradation if unavailable
        self.redis_client = None
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            import redis
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self.redis_client = client
            log.log_info("[pipeline] Redis connected")
        except Exception as exc:
            log.log_warn(f"[pipeline] Redis unavailable — cancellation disabled: {exc}")

    def finalize(self) -> None:
        import torch
        try:
            del self.model
        except AttributeError:
            pass
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Triton entry-point — receives a batch of 1..max_batch_size requests
    # ------------------------------------------------------------------

    def execute(self, requests):
        # Extract all inputs upfront
        batch = [self._extract(req) for req in requests]

        # Run as a single batched generate() call
        try:
            texts = self._infer_batch(
                [b["prompt"]     for b in batch],
                [b["image_b64"]  for b in batch],
                [b["request_id"] for b in batch],
            )
        except Exception as exc:
            import traceback
            pb_utils.Logger.log_error(
                f"[pipeline] Batch inference error: {exc}\n{traceback.format_exc()}"
            )
            texts = [""] * len(batch)

        responses = []
        for text, item in zip(texts, batch):
            text = self._clean_output(text, item["prompt"])
            out  = pb_utils.Tensor("TEXT", np.array([text], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    # ------------------------------------------------------------------
    # Input extraction helper
    # ------------------------------------------------------------------

    def _extract(self, request) -> dict:
        def _str(name: str) -> str:
            t = pb_utils.get_input_tensor_by_name(request, name)
            if t is None:
                return ""
            return _to_str(t.as_numpy().reshape(-1)[0])

        image_b64 = _str("IMAGE_B64")
        if not image_b64.strip():
            raise ValueError("IMAGE_B64 must be provided")

        return {
            "prompt":     _str("PROMPT"),
            "image_b64":  image_b64,
            "request_id": _str("REQUEST_ID"),
        }

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def _infer_batch(self, prompts: list, image_b64s: list, request_ids: list) -> list:
        import torch
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList

        batch_size = len(prompts)

        # Decode images
        images = [
            Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
            for b in image_b64s
        ]

        # Build per-item chat messages
        all_messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  prompt},
            ]}]
            for prompt, img in zip(prompts, images)
        ]

        # Apply chat template to each item
        text_inputs = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]

        # Tokenise + encode all images together (left-padded for batching)
        try:
            from qwen_vl_utils import process_vision_info
            all_image_inputs = []
            for msgs in all_messages:
                img_inp, _ = process_vision_info(msgs)
                if img_inp:
                    all_image_inputs.extend(img_inp)
            inputs = self.processor(
                text=text_inputs,
                images=all_image_inputs if all_image_inputs else None,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            inputs = self.processor(
                text=text_inputs,
                images=images,
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.device)

        # ── Redis-driven stopping criteria ────────────────────────────
        # Stops when ALL requests in the batch are cancelled.
        cancel_flags = [threading.Event() for _ in range(batch_size)]
        redis_client = self.redis_client

        class _BatchCancelCriteria(StoppingCriteria):
            def __init__(self):
                self._n = 0

            def __call__(self, input_ids, scores, **_):
                self._n += 1
                if redis_client and self._n % 5 == 0:
                    for flag, req_id in zip(cancel_flags, request_ids):
                        if req_id and not flag.is_set():
                            try:
                                if redis_client.exists(f"cancel:{req_id}"):
                                    flag.set()
                            except Exception:
                                pass
                return all(f.is_set() for f in cancel_flags)

        # ── Generate (whole batch in one call) ────────────────────────
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([_BatchCancelCriteria()]),
            )

        # ── Decode only the newly generated tokens for each item ──────
        prompt_len = inputs["input_ids"].shape[1]   # same for all (left-padded)
        results = []
        for gen_ids, flag in zip(generated_ids, cancel_flags):
            if flag.is_set():
                results.append("")
                continue
            new_ids = gen_ids[prompt_len:]
            text = self.processor.decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.append(text)

        return results

    # ------------------------------------------------------------------
    # Output cleaning
    # ------------------------------------------------------------------

    def _clean_output(self, text: str, user_prompt: str) -> str:
        for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"):
            text = text.replace(marker, "")
        text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)
        if user_prompt and text.startswith(user_prompt):
            text = text[len(user_prompt):]
        return text.strip()
