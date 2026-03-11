import http.client
import json
import os
import re
from urllib.parse import urlparse

import numpy as np
import redis
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

        self.generate_url = f"{self.triton_http_url}/v2/models/{self.engine_model_name}/generate_stream"
        self.max_tokens   = int(params.get("max_tokens", {}).get("string_value", "4096"))

        redis_url    = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis  = redis.Redis.from_url(redis_url, decode_responses=True)

    def _build_raw_prompt(self, prompt: str) -> str:
        return (
            f"<|im_start|>user\n"
            f"<|img|><|imgpad|><|endofimg|>"
            f"{prompt}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _clean_output(self, text: str, user_prompt: str) -> str:
        marker = "<|im_start|>assistant\n"
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
        else:
            text = re.sub(
                r"^(?:<\|img\|>)?(?:<\|imgpad\|>)*<\|endofimg\|>\s*",
                "",
                text,
                flags=re.DOTALL,
            )
            prompt_pat = r"^\s*" + re.escape(user_prompt) + r"\s*"
            text = re.sub(prompt_pat, "", text, count=1, flags=re.DOTALL)

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
                "stream": True,          # streaming enables token-level cancel checks
                "temperature": 0.05,
                "top_p": 0.9,
                "max_tokens": self.max_tokens,
                "repetition_penalty": 1.2,
                "structured_outputs": json.dumps({"json": self._JSON_SCHEMA})
            }
        }

        body   = json.dumps(payload).encode("utf-8")
        parsed = urlparse(self.generate_url)
        conn   = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=300)

        try:
            conn.request("POST", parsed.path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()

            if resp.status != 200:
                detail = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Engine HTTPError {resp.status}: {detail}")

            # Read NDJSON stream token by token.
            # Check Redis cancel key every 5 tokens to keep overhead low.
            last_output = ""
            buf         = b""
            token_count = 0

            while True:
                chunk = resp.read(512)
                if not chunk:
                    break
                buf += chunk

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    token_count += 1
                    if request_id and token_count % 5 == 0:
                        if self._redis.exists(f"cancel:{request_id}"):
                            self._redis.delete(f"cancel:{request_id}")
                            conn.close()
                            raise RuntimeError("Cancelled")

                    if line.startswith(b"data: "):
                        line = line[6:]
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in obj:
                        raise RuntimeError(f"Engine error: {obj['error']}")
                    if "text_output" in obj:
                        last_output += obj["text_output"]

        finally:
            conn.close()

        if not last_output:
            raise RuntimeError("Engine returned no output")

        return last_output

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                prompt_tensor    = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                image_b64_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_B64")

                if prompt_tensor is None:
                    raise ValueError("Missing input tensor: PROMPT")
                if image_b64_tensor is None:
                    raise ValueError("Missing input tensor: IMAGE_B64")

                prompt    = _to_str(prompt_tensor.as_numpy().reshape(-1)[0])
                image_b64 = _to_str(image_b64_tensor.as_numpy().reshape(-1)[0])

                if not image_b64.strip():
                    raise ValueError("IMAGE_B64 must be provided")

                request_id_tensor = pb_utils.get_input_tensor_by_name(request, "REQUEST_ID")
                request_id = _to_str(request_id_tensor.as_numpy().reshape(-1)[0]) if request_id_tensor is not None else ""

                raw_text   = self._call_engine(prompt, image_b64, request_id)
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
