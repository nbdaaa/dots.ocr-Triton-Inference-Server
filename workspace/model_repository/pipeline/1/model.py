import http.client
import json
import os
from urllib.parse import urlparse

import numpy as np
import redis
import triton_python_backend_utils as pb_utils


def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _image_media_type(image_b64: str) -> str:
    if image_b64.startswith("/9j/"):
        return "image/jpeg"
    if image_b64.startswith("iVBOR"):
        return "image/png"
    if image_b64.startswith("R0lGOD"):
        return "image/gif"
    if image_b64.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        self.engine_model_name = params.get("engine_model_name", {}).get("string_value", "dots_mocr")

        triton_http_port = os.environ.get("TRITON_HTTP_PORT")
        if triton_http_port:
            self.triton_http_url = f"http://127.0.0.1:{triton_http_port}"
        else:
            self.triton_http_url = params.get("triton_http_url", {}).get("string_value", "http://127.0.0.1:8000")

        self.chat_url  = f"{self.triton_http_url}/v1/chat/completions"
        self.max_tokens = int(params.get("max_tokens", {}).get("string_value", "4096"))

        redis_url   = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def _call_engine(self, prompt: str, image_b64: str, request_id: str = "") -> str:
        media_type = _image_media_type(image_b64)
        payload = {
            "model": self.engine_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_b64}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        body   = json.dumps(payload).encode("utf-8")
        parsed = urlparse(self.chat_url)
        conn   = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=300)

        try:
            conn.request("POST", parsed.path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()

            if resp.status != 200:
                detail = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Engine HTTPError {resp.status}: {detail}")

            # SSE stream: lines of "data: {...}" or "data: [DONE]"
            # Check Redis cancel key every 5 tokens to keep overhead low.
            output      = ""
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
                    if not line or not line.startswith(b"data: "):
                        continue

                    data = line[6:]
                    if data == b"[DONE]":
                        break

                    token_count += 1
                    if request_id and token_count % 5 == 0:
                        if self._redis.exists(f"cancel:{request_id}"):
                            self._redis.delete(f"cancel:{request_id}")
                            conn.close()
                            raise RuntimeError("Cancelled")

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if "error" in obj:
                        raise RuntimeError(f"Engine error: {obj['error']}")

                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        output += delta["content"]

        finally:
            conn.close()

        if not output:
            raise RuntimeError("Engine returned no output")

        return output

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

                text = self._call_engine(prompt, image_b64, request_id)

                out_tensor = pb_utils.Tensor(
                    "TEXT",
                    np.array([text], dtype=object),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                )

        return responses
