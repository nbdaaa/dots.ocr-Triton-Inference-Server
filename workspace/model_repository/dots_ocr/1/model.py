import json
import os
import re
import urllib.request
import urllib.error
import base64

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

        self.engine_model_name = params.get("engine_model_name", {}).get("string_value", "dots_ocr_engine")

        triton_http_port = os.environ.get("TRITON_HTTP_PORT")
        if triton_http_port:
            self.triton_http_url = f"http://127.0.0.1:{triton_http_port}"
        else:
            self.triton_http_url = params.get("triton_http_url", {}).get("string_value", "http://127.0.0.1:8000")

        self.generate_url = f"{self.triton_http_url}/v2/models/{self.engine_model_name}/generate"

    def _build_raw_prompt(self, prompt: str) -> str:
        return f"<|img|><|imgpad|><|endofimg|>\n{prompt}"

    def _clean_output(self, text: str, user_prompt: str) -> str:
        text = re.sub(
            r"^<\|img\|>(?:<\|imgpad\|>)+<\|endofimg\|>\s*",
            "",
            text,
            flags=re.DOTALL,
        )

        prompt_pat = r"^\s*" + re.escape(user_prompt) + r"\s*"
        text = re.sub(prompt_pat, "", text, count=1, flags=re.DOTALL)

        return text.strip()

    def _fetch_image_as_b64(self, image_url: str) -> str:
        if not image_url.startswith(("http://", "https://")):
            raise ValueError("IMAGE_URL must start with http:// or https://")

        req = urllib.request.Request(
            image_url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                img_bytes = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to fetch IMAGE_URL (HTTP {e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch IMAGE_URL: {e}") from e

        if not img_bytes:
            raise ValueError("IMAGE_URL returned empty content")

        return base64.b64encode(img_bytes).decode("utf-8")

    def _resolve_image_b64(self, image_b64: str, image_url: str) -> str:
        image_b64 = (image_b64 or "").strip()
        image_url = (image_url or "").strip()

        # ưu tiên IMAGE_URL nếu có
        if image_url:
            return self._fetch_image_as_b64(image_url)

        if image_b64:
            return image_b64

        raise ValueError("Either IMAGE_URL or IMAGE_B64 must be provided")

    def _call_engine(self, prompt: str, image_b64: str) -> str:
        payload = {
            "text_input": self._build_raw_prompt(prompt),
            "image": [image_b64],
            "parameters": {
                "stream": False,
                "temperature": 0,
                "max_tokens": 1024
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

        return resp_json["text_output"]

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                image_b64_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_B64")
                image_url_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_URL")

                if prompt_tensor is None:
                    raise ValueError("Missing input tensor: PROMPT")
                if image_b64_tensor is None:
                    raise ValueError("Missing input tensor: IMAGE_B64")
                if image_url_tensor is None:
                    raise ValueError("Missing input tensor: IMAGE_URL")

                prompt = _to_str(prompt_tensor.as_numpy().reshape(-1)[0])
                image_b64_in = _to_str(image_b64_tensor.as_numpy().reshape(-1)[0])
                image_url = _to_str(image_url_tensor.as_numpy().reshape(-1)[0])

                resolved_b64 = self._resolve_image_b64(image_b64_in, image_url)
                raw_text = self._call_engine(prompt, resolved_b64)
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
