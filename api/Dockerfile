FROM python:3.12-slim
WORKDIR /app
RUN pip install fastapi uvicorn python-multipart pymupdf --no-cache-dir
COPY main.py .
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${API_PORT:-8080}"]
