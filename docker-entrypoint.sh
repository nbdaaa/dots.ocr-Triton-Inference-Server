#!/bin/bash
set -e

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[entrypoint] No GPUs detected, falling back to 1 CPU instance"
    KIND="KIND_CPU"
else
    echo "[entrypoint] Detected $NUM_GPUS GPU(s) → launching 1 instance per GPU ($NUM_GPUS total)"
    KIND="KIND_GPU"
fi

# Overwrite dots_ocr config.pbtxt.
# count: 1 + KIND_GPU → Triton creates exactly 1 instance per available GPU.
# Do NOT set count to NUM_GPUS — that would create NUM_GPUS instances *per* GPU.
cat > /models/dots_ocr/config.pbtxt << EOF
backend: "vllm"
instance_group [
  {
    count: 1
    kind: ${KIND}
  }
]
EOF

echo "[entrypoint] config.pbtxt updated: count=1, kind=${KIND} (auto-scaled across $NUM_GPUS GPU(s))"

# Start Triton
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
