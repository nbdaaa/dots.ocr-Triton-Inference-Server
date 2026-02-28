# README

## Chạy và test Triton Inference Server cho `dots_ocr`

Tài liệu này hướng dẫn cách:

1. Khởi chạy Triton Inference Server với model repository đã chuẩn bị sẵn
2. Gửi request test OCR bằng `curl` với ảnh từ URL

## Yêu cầu trước khi chạy

* Đã cài **Docker**
* Đã cài **NVIDIA Container Toolkit** để dùng được `--gpus all`
* Đã chuẩn bị cấu trúc thư mục như sau:

```text
└── workspace
    ├── duthao.png
    ├── model_repository
        ├── dots_ocr
        │   ├── 1
        │   │   └── model.py
        │   └── config.pbtxt
        └── dots_ocr_engine
            ├── 1
            │   └── model.json
            └── config.pbtxt
```

* Thư mục cache Hugging Face đã tồn tại (nếu có) tại:

```text
~/.cache/huggingface
```

## 1. Khởi chạy Triton Server

Chạy lệnh sau trong terminal thứ nhất:

```bash
docker run --rm --gpus all \
  --network host \
  --ipc=host \
  -v /workspace/model_repository:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/tritonserver:25.11-vllm-python-py3 \
  tritonserver \
    --model-repository=/models \
    --http-port=54280 \
    --grpc-port=8001 \
    --metrics-port=8002
```

### Giải thích nhanh

* `--gpus all`: cho container dùng GPU
* `--network host`: dùng network của máy host
* `--ipc=host`: chia sẻ IPC để tối ưu hiệu năng
* `-v /workspace/model_repository:/models`: mount model repository vào container
* `-v ~/.cache/huggingface:/root/.cache/huggingface`: tái sử dụng cache model
* `--http-port=54280`: Triton nhận HTTP request ở cổng `54280`

> Giữ terminal này mở trong suốt quá trình test.

## 2. Test model bằng `curl`

Mở terminal thứ hai và chạy:

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

## 3. Ý nghĩa request test

Request trên gửi vào model `dots_ocr` với 3 input:

* `PROMPT`: chỉ dẫn OCR cho model
* `IMAGE_B64`: để trống, vì ở đây không dùng ảnh base64
* `IMAGE_URL`: URL ảnh để wrapper server tự tải ảnh và xử lý

## 4. Kết quả mong đợi

Server sẽ trả về JSON dạng:

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

Nội dung OCR nằm tại:

```text
outputs[0].data[0]
```

## 5. Kiểm tra nhanh server đã sẵn sàng chưa

Nếu muốn kiểm tra trước khi test OCR:

```bash
curl http://127.0.0.1:54280/v2/health/ready
```

Nếu server sẵn sàng, bạn sẽ nhận phản hồi thành công.


