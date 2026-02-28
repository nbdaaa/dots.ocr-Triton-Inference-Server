# README

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
├── dots_ocr/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── dots_ocr_engine/
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

* `dots_ocr`
* `dots_ocr_engine`

---

## 5. Test OCR bằng `curl` với ảnh URL

Chạy lệnh sau:

```bash
curl -s -X POST http://127.0.0.1:54280/v2/models/dots_ocr/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "PROMPT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["OCR this image and return markdown."]
      },
      {
        "name": "IMAGE_B64",
        "shape": [1],
        "datatype": "BYTES",
        "data": [""]
      },
      {
        "name": "IMAGE_URL",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["https://cdn.thuvienphapluat.vn//uploads/tintuc/2022/07/19/du-thao.png"]
      }
    ]
  }'
```

### Giải thích request

Wrapper `dots_ocr` nhận 3 input:

* `PROMPT`: chỉ dẫn cho model (Không cần thiết, có thể bỏ trống)
* `IMAGE_B64`: để trống nếu không dùng base64
* `IMAGE_URL`: URL ảnh để server tự tải ảnh và OCR

---

## 6. Kết quả trả về

Server sẽ trả JSON dạng:

```json
{
  "model_name": "dots_ocr",
  "model_version": "1",
  "outputs": [
    {
      "name": "TEXT",
      "datatype": "BYTES",
      "shape": [1],
      "data": [
        "..."
      ]
    }
  ]
}
```

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
