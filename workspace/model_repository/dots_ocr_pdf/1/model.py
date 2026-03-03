import base64
import json
import os
import urllib.request
import urllib.error

import numpy as np
import triton_python_backend_utils as pb_utils


DEFAULT_PROMPT = "OCR this image and return the text content."
DEFAULT_DPI = 200


def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        self.ocr_model_name = params.get("ocr_model_name", {}).get("string_value", "dots_ocr")

        triton_http_port = os.environ.get("TRITON_HTTP_PORT")
        if triton_http_port:
            self.triton_http_url = f"http://127.0.0.1:{triton_http_port}"
        else:
            self.triton_http_url = params.get("triton_http_url", {}).get("string_value", "http://127.0.0.1:8000")

        self.infer_url = f"{self.triton_http_url}/v2/models/{self.ocr_model_name}/infer"
        self.dpi = int(params.get("dpi", {}).get("string_value", str(DEFAULT_DPI)))

        try:
            import fitz
            self._fitz = fitz
        except ImportError:
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF processing. "
                "Install it with: pip install pymupdf"
            )

    def _pdf_to_page_images(self, pdf_source) -> list:
        """Render each PDF page as a base64-encoded PNG string.
        pdf_source: file path (str) or raw PDF bytes.
        """
        mat = self._fitz.Matrix(self.dpi / 72, self.dpi / 72)
        page_images = []
        open_kwargs = {"stream": pdf_source, "filetype": "pdf"} if isinstance(pdf_source, bytes) else {"filename": pdf_source}
        with self._fitz.open(**open_kwargs) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                png_bytes = pix.tobytes("png")
                page_images.append(base64.b64encode(png_bytes).decode("utf-8"))
        return page_images

    def _call_ocr(self, prompt: str, image_b64: str) -> str:
        """Send a single image to dots_ocr/infer and return the text."""
        payload = {
            "inputs": [
                {"name": "PROMPT",    "shape": [1], "datatype": "BYTES", "data": [prompt]},
                {"name": "IMAGE_B64", "shape": [1], "datatype": "BYTES", "data": [image_b64]},
                {"name": "IMAGE_URL", "shape": [1], "datatype": "BYTES", "data": [""]},
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.infer_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"dots_ocr HTTPError {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"dots_ocr URLError: {e}") from e

        if "error" in resp_json:
            raise RuntimeError(resp_json["error"])

        for out in resp_json.get("outputs", []):
            if out.get("name") == "TEXT":
                data = out.get("data", [])
                return data[0] if data else ""

        raise RuntimeError(f"Unexpected response from dots_ocr: {resp_json}")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                pdf_path_tensor = pb_utils.get_input_tensor_by_name(request, "PDF_PATH")
                pdf_b64_tensor  = pb_utils.get_input_tensor_by_name(request, "PDF_B64")
                prompt_tensor   = pb_utils.get_input_tensor_by_name(request, "PROMPT")

                if prompt_tensor is None:
                    raise ValueError("Missing input tensor: PROMPT")

                prompt   = _to_str(prompt_tensor.as_numpy().reshape(-1)[0]).strip() or DEFAULT_PROMPT
                pdf_path = _to_str(pdf_path_tensor.as_numpy().reshape(-1)[0]).strip() if pdf_path_tensor is not None else ""
                pdf_b64  = _to_str(pdf_b64_tensor.as_numpy().reshape(-1)[0]).strip()  if pdf_b64_tensor  is not None else ""

                if pdf_b64:
                    pdf_source = base64.b64decode(pdf_b64)
                elif pdf_path:
                    if not os.path.isfile(pdf_path):
                        raise ValueError(f"PDF file not found: {pdf_path}")
                    pdf_source = pdf_path
                else:
                    raise ValueError("Either PDF_B64 or PDF_PATH must be provided")

                page_images = self._pdf_to_page_images(pdf_source)
                if not page_images:
                    raise ValueError(f"PDF has no renderable pages: {pdf_path}")

                page_texts = []
                for i, img_b64 in enumerate(page_images):
                    text = self._call_ocr(prompt, img_b64)
                    page_texts.append(f"[Page {i + 1}]\n{text}")

                full_text = "\n\n".join(page_texts)

                out_tensor = pb_utils.Tensor("TEXT", np.array([full_text], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )

        return responses
