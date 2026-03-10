import json
import os
import re
import urllib.request
import urllib.error

import numpy as np
import triton_python_backend_utils as pb_utils


def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        self.engine_model_name = params.get("engine_model_name", {}).get("string_value", "dots_ocr")

        triton_http_port = os.environ.get("TRITON_HTTP_PORT")
        if triton_http_port:
            self.triton_http_url = f"http://127.0.0.1:{triton_http_port}"
        else:
            self.triton_http_url = params.get("triton_http_url", {}).get("string_value", "http://127.0.0.1:8000")

        self.generate_url = f"{self.triton_http_url}/v2/models/{self.engine_model_name}/generate"
        self.max_tokens = int(params.get("max_tokens", {}).get("string_value", "4096"))

    def _build_raw_prompt(self, prompt: str) -> str:
        return (
            f"<|im_start|>user\n"
            f"<|img|><|imgpad|><|endofimg|>"
            f"{prompt}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _clean_output(self, text: str, user_prompt: str) -> str:
        # text_output from vLLM includes the full sequence (input + generated).
        # Extract only the assistant's generated portion.
        marker = "<|im_start|>assistant\n"
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
        else:
            # Fallback: strip image tokens then echoed prompt
            text = re.sub(
                r"^(?:<\|img\|>)?(?:<\|imgpad\|>)*<\|endofimg\|>\s*",
                "",
                text,
                flags=re.DOTALL,
            )
            prompt_pat = r"^\s*" + re.escape(user_prompt) + r"\s*"
            text = re.sub(prompt_pat, "", text, count=1, flags=re.DOTALL)

        # Strip trailing end-of-turn token if present
        text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)

        return text.strip()

    _JSON_SCHEMA = json.dumps({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "Caption", "Footnote", "Formula", "List-item",
                        "Page-footer", "Page-header", "Picture",
                        "Section-header", "Table", "Text", "Title"
                    ]
                },
                "text": {"type": "string"}
            },
            "required": ["bbox", "category"]
        }
    })

    def _call_engine(self, prompt: str, image_b64: str, request_id: str = "") -> str:
        payload = {
            "text_input": self._build_raw_prompt(prompt),
            "image": [image_b64],
            "parameters": {
                "stream": False,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": self.max_tokens,
                "repetition_penalty": 1.1,
                "structured_outputs": json.dumps({"json": self._JSON_SCHEMA})
            }
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.generate_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Engine HTTPError {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Engine URLError: {e}") from e

        if "error" in resp_json:
            raise RuntimeError(resp_json["error"])

        if "text_output" not in resp_json:
            raise RuntimeError(f"Unexpected engine response: {resp_json}")

        raw = resp_json["text_output"]
        print(f"[DEBUG raw_output] {repr(raw[:])}", flush=True)
        return raw

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                image_b64_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_B64")

                if prompt_tensor is None:
                    raise ValueError("Missing input tensor: PROMPT")
                if image_b64_tensor is None:
                    raise ValueError("Missing input tensor: IMAGE_B64")

                prompt = _to_str(prompt_tensor.as_numpy().reshape(-1)[0])
                image_b64 = _to_str(image_b64_tensor.as_numpy().reshape(-1)[0])

                if not image_b64.strip():
                    raise ValueError("IMAGE_B64 must be provided")

                request_id_tensor = pb_utils.get_input_tensor_by_name(request, "REQUEST_ID")
                request_id = _to_str(request_id_tensor.as_numpy().reshape(-1)[0]) if request_id_tensor is not None else ""

                raw_text = self._call_engine(prompt, image_b64, request_id)
                clean_text = self._clean_output(raw_text, prompt)

                out_tensor = pb_utils.Tensor(
                    "TEXT",
                    np.array([clean_text], dtype=object),
                )

                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                )

        return responses
