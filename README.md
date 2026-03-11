# README

## Monitoring: Prometheus + Grafana

Triton tự động expose metrics Prometheus tại cổng `8002`. Stack monitoring được khởi động cùng Triton bằng một lệnh duy nhất.

### Yêu cầu

* Docker Engine >= 20.10 và Docker Compose v2 (plugin `docker compose`)
* NVIDIA Container Toolkit đã cài và cấu hình

### Bước 1 — Tải dashboard JSON (chỉ làm một lần)

```bash
curl -fsSL "https://grafana.com/api/dashboards/22897/revisions/latest/download" \
  -o monitoring/grafana/dashboards/triton-inference-server.json
```

### Bước 2 — Khởi động toàn bộ stack

Trước khi chạy hệ thống, hãy tạo file `.env` ở thư mục gốc của project. Cấu trúc file `.env`:
```env
# Triton Inference Server
TRITON_IMAGE_TAG=25.11-vllm-python-py3
TRITON_HTTP_PORT= HTTP PORT of Server
TRITON_GRPC_PORT=8001
TRITON_METRICS_PORT=8002
API_PORT=12345
PDF_INPUT_PATH=/home/user/pdfs
MODEL_REPO_PATH=./workspace/model_repository
HF_CACHE_PATH=~/.cache/huggingface

# Prometheus
PROMETHEUS_IMAGE_TAG=latest
PROMETHEUS_PORT=9090
PROMETHEUS_RETENTION=15d

# Grafana
GRAFANA_IMAGE_TAG=latest
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# Redis
REDIS_IMAGE_TAG=7-alpine
REDIS_PORT=6379

# RedisInsight
REDISINSIGHT_PORT=5540
```

Sau khi tạo file `.env`, chạy docker:s

```bash
docker compose up -d
```

Ba container sẽ khởi động: **triton**, **prometheus**, **grafana**.

### Truy cập

| Dịch vụ      | URL                        | Ghi chú              |
|---------------|----------------------------|----------------------|
| Triton        | http://localhost:54280     | HTTP inference API   |
| API           | http://localhost:     | FastAPI endpoints    |
| Prometheus    | http://localhost:9090      | targets → triton UP  |
| Grafana       | http://localhost:3000      | admin / admin        |
| RedisInsight  | http://localhost:5540      | Redis GUI            |

```bash
ssh -p PORT_SSH root@IP_PUBLIC -L 8080:localhost:8080 -L 3000:localhost:3000 -L 9090:localhost:9090 -L 5540:localhost:5540 -L HTTP_PORT:localhost:HTTP_PORT
```

Trong Grafana: vào **Dashboards → Triton Inference Server** để xem dashboard NVIDIA Triton.

### Dừng stack

```bash
docker compose down
```

---

## Chạy và test Triton Inference Server cho `dots_ocr`

Tài liệu này hướng dẫn cách:

1. Khởi chạy Triton Inference Server với model repository đã chuẩn bị sẵn
2. Test OCR bằng `curl`
3. Khắc phục lỗi Docker/NVIDIA runtime thường gặp

---

## 1. Yêu cầu trước khi chạy

* Đã cài **Docker**
* Đã cài **NVIDIA Container Toolkit**
* Host đã nhận GPU (`nvidia-smi` chạy được)
* **NVIDIA Driver version phải >= `580.95.05`**

* Model repository nằm tại:

```text
workspace/model_repository
```

* Cấu trúc thư mục:

```text
workspace/model_repository/
├── pipeline/          # Python backend — nhận request, gọi dots_ocr
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── dots_ocr/          # vLLM backend — chạy model rednote-hilab/dots.ocr
    ├── config.pbtxt
    └── 1/
        └── model.json
```

---

## 2. Kiểm tra nhanh driver version

Chạy:

```bash
nvidia-smi
```

hoặc lấy riêng version:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

Nếu kết quả **nhỏ hơn `580.95.05`**, không nên dùng image `25.11-vllm-python-py3`. ([NVIDIA Docs][1])

---

## 3. Khởi chạy Triton Server

Chạy lệnh sau trong terminal thứ nhất:

```bash
HTTP_PORT=54280

docker run --rm --gpus all \
  --network host \
  --ipc=host \
  -v /home/workspace/model_repository:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e TRITON_HTTP_PORT=$HTTP_PORT \
  nvcr.io/nvidia/tritonserver:25.11-vllm-python-py3 \
  tritonserver \
    --model-repository=/models \
    --http-port=$HTTP_PORT \
    --grpc-port=8001 \
    --metrics-port=8002
```

### Giải thích nhanh

* `--gpus all`: cho container dùng GPU
* `--network host`: dùng network của host
* `--ipc=host`: chia sẻ IPC
* `-v /home/workspace/model_repository:/models`: mount model repository vào container
* `-v ~/.cache/huggingface:/root/.cache/huggingface`: tái sử dụng cache model
* `HTTP_PORT=54280`: đặt biến shell một lần duy nhất — cả `--http-port` lẫn internal call của `dots_ocr` → `dots_ocr_engine` đều dùng cổng này
* `-e TRITON_HTTP_PORT=$HTTP_PORT`: truyền cổng vào container để `dots_ocr` tự cấu hình URL nội bộ

> Giữ terminal này mở trong suốt quá trình test.

---

## 4. Kiểm tra server đã sẵn sàng chưa

Mở terminal thứ hai:

```bash
curl http://127.0.0.1:54280/v2/health/ready
```

Nếu server sẵn sàng, bạn sẽ nhận phản hồi thành công.

Kiểm tra model đã được load:

```bash
curl -s http://127.0.0.1:54280/v2/repository/index
```

Bạn nên thấy:

* `pipeline`
* `dots_ocr`

---

## 5. Test Triton pipeline trực tiếp bằng `curl`

```bash
IMAGE_B64=$(base64 -w 0 /path/to/image.png)

curl -s -X POST http://127.0.0.1:54280/v2/models/pipeline/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": [
      {
        \"name\": \"PROMPT\",
        \"shape\": [1],
        \"datatype\": \"BYTES\",
        \"data\": [\"\"]
      },
      {
        \"name\": \"IMAGE_B64\",
        \"shape\": [1],
        \"datatype\": \"BYTES\",
        \"data\": [\"${IMAGE_B64}\"]
      }
    ]
  }"
```

### Giải thích request

Model `pipeline` nhận 2 input:

* `PROMPT`: chỉ dẫn cho model (có thể bỏ trống)
* `IMAGE_B64`: ảnh đã encode base64

---

## 6. Kết quả trả về từ Triton

Server sẽ trả JSON dạng:

```json
{
  "model_name": "pipeline",
  "model_version": "1",
  "outputs": [
    {
      "name": "TEXT",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["..."]
    }
  ]
}
```

---

## 7. API Endpoints (FastAPI)

API chạy tại cổng `API_PORT` (mặc định `54188`), đóng vai trò trung gian giữa client và Triton `pipeline`.

### POST `/infer-image`

Upload một ảnh và nhận kết quả OCR.

**Request** — `multipart/form-data`:

| Field    | Type   | Bắt buộc | Mô tả                        |
|----------|--------|-----------|------------------------------|
| `file`   | file   | Có        | File ảnh (PNG, JPG, ...)     |
| `prompt` | string | Không     | Chỉ dẫn cho model            |

```bash
curl -s -X POST http://localhost:54188/infer-image \
  -F "file=@/path/to/image.png" \
  -F "prompt="
```

**Response**:

```json
{ "text": "..." }
```

---

### POST `/infer-pdf`

Upload một file PDF mới, render từng trang thành ảnh và OCR song song tất cả các trang. PDF gốc được lưu lại trên server để hỗ trợ resume sau này.

**Request** — `multipart/form-data`:

| Field    | Type   | Bắt buộc | Mô tả             |
|----------|--------|-----------|-------------------|
| `file`   | file   | Có        | File PDF          |
| `prompt` | string | Không     | Chỉ dẫn cho model |

```bash
curl -s -X POST http://localhost:54188/infer-pdf \
  -F "file=@/path/to/document.pdf"
```

**Response** — NDJSON stream:

```
{"job_id": "uuid", "status": "processing"}
{"file_path": "document.pdf", "filename": "document", "page_idx": 0, "image_path": "document_page_0000.png", "response": "..."}
{"file_path": "document.pdf", "filename": "document", "page_idx": 2, "image_path": "document_page_0002.png", "response": "..."}
...
{"job_id": "uuid", "status": "completed"}
```

> Lưu ý: các trang không nhất thiết trả về theo thứ tự do được xử lý song song.

---

### POST `/pause-pdf/{job_id}`

Dừng một job đang xử lý. Các trang đã hoàn thành được giữ nguyên trong Redis. Các trang đang generate token sẽ bị huỷ trong vòng 5 token. Các trang chưa bắt đầu bị huỷ ngay lập tức.

```bash
curl -s -X POST http://localhost:54188/pause-pdf/<job_id>
```

**Response**:

```json
{
  "job_id": "uuid",
  "status": "paused",
  "pages_saved": 5,
  "pages_remaining": 6
}
```

---

### POST `/resume-pdf/{job_id}`

Tiếp tục một job đang ở trạng thái `paused` hoặc `failed`. Server tự đọc lại PDF từ disk, chỉ xử lý các trang chưa hoàn thành.

```bash
curl -s -X POST http://localhost:54188/resume-pdf/<job_id>
```

**Response** — NDJSON stream (giống `/infer-pdf`):

```
{"job_id": "uuid", "status": "processing"}
{"file_path": "document.pdf", "filename": "document", "page_idx": 5, ...}
...
{"job_id": "uuid", "status": "completed"}
```

Trả `400` nếu job không ở trạng thái `paused` hoặc `failed`.

---

### DELETE `/delete-pdf/{job_id}`

Xoá hoàn toàn một job khỏi Redis và xoá file PDF đã lưu trên disk. Job phải ở trạng thái `paused`, `completed` hoặc `failed` (không thể xoá job đang `processing`).

```bash
curl -s -X DELETE http://localhost:54188/delete-pdf/<job_id>
```

**Response**:

```json
{
  "job_id": "uuid",
  "deleted": true
}
```

Trả `400` nếu job đang `processing`. Pause trước, sau đó mới xoá.

---

### GET `/pdf-status`

Liệt kê trạng thái tất cả các job.

```bash
curl -s http://localhost:54188/pdf-status
```

**Response**:

```json
[
  {
    "job_id": "uuid",
    "status": "processing | completed | paused | failed",
    "filename": "document.pdf",
    "ocr_total_pages": 5,
    "ocr_processed_pages": 3,
    "ocr_success_pages": 2,
    "ocr_fail_pages": 1,
    "ocr_remaining_pages": 2
  }
]
```

---

### GET `/pdf-status/{job_id}`

Lấy trạng thái và kết quả của một job cụ thể.

```bash
curl -s http://localhost:54188/pdf-status/<job_id>
```

**Response** (khi `completed`):

```json
{
  "job_id": "uuid",
  "status": "completed",
  "filename": "document.pdf",
  "ocr_total_pages": 5,
  "ocr_processed_pages": 5,
  "ocr_success_pages": 5,
  "ocr_fail_pages": 0,
  "ocr_remaining_pages": 0,
  "text": "[Page 1]\n...\n\n[Page 2]\n..."
}
```

Trả `404` nếu `job_id` không tồn tại.

## 7. Khắc phục lỗi Docker chưa được cấu hình cho NVIDIA runtime

Nếu khi chạy Triton với `--gpus all` bạn gặp lỗi kiểu:

```text
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

thì nguyên nhân thường là **Docker daemon chưa được cấu hình để dùng NVIDIA runtime**.
Sau khi cài NVIDIA Container Toolkit, bạn cần chạy `nvidia-ctk runtime configure --runtime=docker`; lệnh này sửa `/etc/docker/daemon.json`, rồi bạn phải restart Docker. ([NVIDIA Docs][2])

### Các bước sửa lỗi

```bash
# 1) kiểm tra host đã thấy GPU chưa
nvidia-smi

# 2) cài NVIDIA Container Toolkit (nếu chưa cài)
apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.2-1
apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# 3) cấu hình Docker dùng NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker

# 4) restart Docker
systemctl restart docker
# nếu máy không có systemd thì thử:
# service docker restart
```

### Kiểm tra Docker đã nhận NVIDIA runtime chưa

Test nhanh GPU trong Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Nếu lệnh này chạy được và in ra thông tin GPU, nghĩa là `--gpus all` đã hoạt động bình thường.

---

[1]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html?utm_source=chatgpt.com "Release Compatibility Matrix — NVIDIA Triton Inference ..."
[2]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html?utm_source=chatgpt.com "Installing the NVIDIA Container Toolkit"
