import asyncio
import base64
import json
import os
import threading
import urllib.request
import urllib.error
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import glob
import subprocess
import tempfile
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse

def JSONResponse(content, **kwargs):
    return Response(
        content=json.dumps(content, ensure_ascii=False),
        media_type="application/json",
        **kwargs,
    )

app = FastAPI()

TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000")
OCR_INFER_URL = f"{TRITON_URL}/v2/models/pipeline/infer"
DPI = 200

DEFAULT_PROMPT = (
    "Please output the layout information from the PDF image, including each layout element's bbox, "
    "its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', "
    "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)

# In-memory job store
jobs: Dict[str, Dict[str, Any]] = {}
thread_pool = ThreadPoolExecutor(max_workers=16)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def render_pdf_pages(pdf_bytes: bytes) -> list:
    pages = []
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        out_prefix = os.path.join(tmpdir, "page")

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        subprocess.run(
            ["pdftoppm", "-png", "-r", str(DPI), pdf_path, out_prefix],
            check=True,
            capture_output=True,
        )

        for img_path in sorted(glob.glob(f"{out_prefix}-*.png")):
            with open(img_path, "rb") as f:
                pages.append(base64.b64encode(f.read()).decode())

    return pages


def ocr_page_sync(image_b64: str, prompt: str, cancel_event: threading.Event, request_id: str) -> str:
    if cancel_event.is_set():
        raise RuntimeError("Cancelled")

    payload = {
        "inputs": [
            {"name": "PROMPT",      "shape": [1], "datatype": "BYTES", "data": [prompt]},
            {"name": "IMAGE_B64",   "shape": [1], "datatype": "BYTES", "data": [image_b64]},
            {"name": "REQUEST_ID",  "shape": [1], "datatype": "BYTES", "data": [request_id]},
        ]
    }
    req = urllib.request.Request(
        OCR_INFER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Triton HTTPError {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Triton URLError: {e}")

    for out in result.get("outputs", []):
        if out["name"] == "TEXT":
            return out["data"][0]
    raise RuntimeError(f"Unexpected Triton response: {result}")


# ─── POST /infer-image ────────────────────────────────────────────────────────

@app.post("/infer-image")
async def infer_image(
    file: UploadFile = File(...),
    prompt: str = Form(default=DEFAULT_PROMPT),
):
    if not prompt:
        prompt = DEFAULT_PROMPT
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(
            thread_pool, ocr_page_sync, image_b64, prompt, threading.Event(), str(uuid.uuid4())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"text": text})


# ─── POST /infer-pdf  (async — streams job_id then result, supports cancel) ──

@app.post("/infer-pdf")
async def infer_pdf(
    file: UploadFile = File(...),
    prompt: str = Form(default=DEFAULT_PROMPT),
):
    if not prompt:
        prompt = DEFAULT_PROMPT
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    job_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    cancel_event = threading.Event()

    # Render pages
    try:
        pages = await loop.run_in_executor(thread_pool, render_pdf_pages, pdf_bytes)
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "filename": file.filename,
            "ocr_total_pages": 0,
            "ocr_processed_pages": 0,
            "ocr_success_pages": 0,
            "ocr_fail_pages": 0,
            "ocr_remaining_pages": 0,
            "error": str(e),
            "cancel_event": cancel_event,
            "page_tasks": [],
        }
        raise HTTPException(status_code=500, detail=f"Failed to render PDF: {e}")

    total = len(pages)
    if total == 0:
        raise HTTPException(status_code=422, detail="PDF has no renderable pages")

    filename_stem = os.path.splitext(file.filename)[0]
    result_queue: asyncio.Queue = asyncio.Queue()

    jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "ocr_total_pages": total,
        "ocr_processed_pages": 0,
        "ocr_success_pages": 0,
        "ocr_fail_pages": 0,
        "ocr_remaining_pages": total,
        "result": None,
        "error": None,
        "cancel_event": cancel_event,
        "page_tasks": [],
    }
    job = jobs[job_id]
    page_results: list = [None] * total

    async def ocr_one(i: int, image_b64: str):
        request_id = str(uuid.uuid4())
        page_result = None
        try:
            text = await loop.run_in_executor(
                thread_pool, ocr_page_sync, image_b64, prompt, cancel_event, request_id
            )
            page_result = {
                "file_path": file.filename,
                "filename": filename_stem,
                "page_idx": i,
                "image_path": f"{filename_stem}_page_{i:04d}.png",
                "response": text,
            }
            page_results[i] = page_result
            job["ocr_success_pages"] += 1
        except asyncio.CancelledError:
            job["ocr_fail_pages"] += 1
            raise
        except Exception as e:
            job["ocr_fail_pages"] += 1
        finally:
            job["ocr_processed_pages"] += 1
            job["ocr_remaining_pages"] -= 1
            result_queue.put_nowait(page_result)

    tasks = [asyncio.create_task(ocr_one(i, img)) for i, img in enumerate(pages)]
    job["page_tasks"] = tasks

    async def stream():
        yield json.dumps({"job_id": job_id, "status": "processing"}, ensure_ascii=False) + "\n"
        for _ in range(total):
            page_result = await result_queue.get()
            if page_result is not None:
                yield json.dumps(page_result, ensure_ascii=False) + "\n"
        job["result"] = [r for r in page_results if r is not None]
        if job["status"] != "cancelled":
            job["status"] = "completed"
        yield json.dumps({"job_id": job_id, "status": job["status"]}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ─── DELETE /infer-pdf/{job_id} ───────────────────────────────────────────────

@app.delete("/infer-pdf/{job_id}")
async def cancel_infer_pdf(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "processing":
        raise HTTPException(status_code=400, detail=f"Job is not processing (status: {job['status']})")

    # Signal all threads to stop before making new Triton calls
    job["cancel_event"].set()
    job["status"] = "cancelled"

    # Cancel asyncio tasks (stops pending pages not yet in the thread pool)
    for task in job.get("page_tasks", []):
        task.cancel()

    return JSONResponse({"job_id": job_id, "status": "cancelled"})


# ─── Status endpoints ─────────────────────────────────────────────────────────

def _job_summary(job_id: str, job: dict, include_text: bool = False) -> dict:
    summary = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job.get("filename", ""),
        "ocr_total_pages": job.get("ocr_total_pages", 0),
        "ocr_processed_pages": job.get("ocr_processed_pages", 0),
        "ocr_success_pages": job.get("ocr_success_pages", 0),
        "ocr_fail_pages": job.get("ocr_fail_pages", 0),
        "ocr_remaining_pages": job.get("ocr_remaining_pages", 0),
    }
    if include_text and job["status"] in ("completed", "cancelled"):
        summary["result"] = job["result"]
    if job["status"] == "failed":
        summary["error"] = job["error"]
    return summary


@app.get("/pdf-status")
async def list_all_status():
    return JSONResponse([
        _job_summary(job_id, job) for job_id, job in jobs.items()
    ])


@app.get("/pdf-status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(_job_summary(job_id, job, include_text=True))
