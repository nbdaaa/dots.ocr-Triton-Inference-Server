import base64
import json
import os
import urllib.request
import urllib.error

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000")
INFER_URL = f"{TRITON_URL}/v2/models/dots_ocr_pdf/infer"


@app.post("/infer-pdf")
async def infer_pdf(
    file: UploadFile = File(...),
    prompt: str = Form(default=""),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode()

    payload = {
        "inputs": [
            {"name": "PDF_B64",  "shape": [1], "datatype": "BYTES", "data": [pdf_b64]},
            {"name": "PDF_PATH", "shape": [1], "datatype": "BYTES", "data": [""]},
            {"name": "PROMPT",   "shape": [1], "datatype": "BYTES", "data": [prompt]},
        ]
    }

    req = urllib.request.Request(
        INFER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Triton error: {detail}")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Triton: {e}")

    text = result["outputs"][0]["data"][0]
    return JSONResponse({"text": text})
