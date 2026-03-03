ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

RUN pip install pymupdf --no-cache-dir
