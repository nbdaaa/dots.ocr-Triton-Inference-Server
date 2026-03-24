"""
dots.ocr — Triton Python backend (pure PyTorch / HuggingFace)

Replaces the former two-model setup (pipeline CPU → dots_ocr vLLM) with a
single model that loads rednote-hilab/dots.ocr directly via HuggingFace
transformers and runs inference on the assigned GPU instance.

External interface is identical to the previous pipeline model:
  Inputs : PROMPT (string [1]), IMAGE_B64 (string [1]), REQUEST_ID (string [1], optional)
  Output : TEXT (string [1])

Cancellation is still driven by Redis: any caller sets  cancel:<request_id>
and generation stops within ~5 tokens.
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

        self.model_name     = _p("model_name", "rednote-hilab/dots.ocr")
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
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
    # Triton entry-point
    # ------------------------------------------------------------------

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._handle(request))
            except Exception as exc:
                import traceback
                pb_utils.Logger.log_error(
                    f"[pipeline] Unhandled error: {exc}\n{traceback.format_exc()}"
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(exc)),
                    )
                )
        return responses

    def _handle(self, request):
        def _str(name: str) -> str:
            t = pb_utils.get_input_tensor_by_name(request, name)
            if t is None:
                return ""
            return _to_str(t.as_numpy().reshape(-1)[0])

        prompt     = _str("PROMPT")
        image_b64  = _str("IMAGE_B64")
        request_id = _str("REQUEST_ID")

        if not image_b64.strip():
            raise ValueError("IMAGE_B64 must be provided")

        text = self._infer(prompt, image_b64, request_id)
        text = self._clean_output(text, prompt)

        out = pb_utils.Tensor("TEXT", np.array([text], dtype=object))
        return pb_utils.InferenceResponse(output_tensors=[out])

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _infer(self, prompt: str, image_b64: str, request_id: str) -> str:
        import torch
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList

        # Decode image
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        # Build Qwen2.5-VL chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  prompt},
                ],
            }
        ]

        # Apply chat template
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenise + encode image pixels
        # Prefer qwen_vl_utils for proper pixel processing; fall back to direct PIL
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            inputs = self.processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.device)

        # Redis-driven stopping criteria
        cancel_flag  = threading.Event()
        redis_client = self.redis_client

        class _CancelCriteria(StoppingCriteria):
            def __init__(self):
                self._n = 0

            def __call__(self, input_ids, scores, **_):
                if cancel_flag.is_set():
                    return True
                self._n += 1
                if request_id and redis_client and self._n % 5 == 0:
                    try:
                        if redis_client.exists(f"cancel:{request_id}"):
                            cancel_flag.set()
                            return True
                    except Exception:
                        pass
                return False

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([_CancelCriteria()]),
            )

        if cancel_flag.is_set():
            return ""

        # Decode only newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_ids    = generated_ids[:, prompt_len:]
        return self.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # ------------------------------------------------------------------
    # Output cleaning
    # ------------------------------------------------------------------

    def _clean_output(self, text: str, user_prompt: str) -> str:
        # Strip any trailing model markers that survived skip_special_tokens
        for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"):
            text = text.replace(marker, "")

        # Strip trailing assistant tag pattern
        text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)

        # Strip any accidental prompt echo
        if user_prompt and text.startswith(user_prompt):
            text = text[len(user_prompt):]

        return text.strip()
