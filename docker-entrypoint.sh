#!/bin/bash
set -e

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[entrypoint] No GPUs detected, falling back to 1 CPU instance"
    NUM_GPUS=1
    KIND="KIND_CPU"
else
    echo "[entrypoint] Detected $NUM_GPUS GPU(s) → launching $NUM_GPUS instance(s)"
    KIND="KIND_GPU"
fi

# Overwrite dots_ocr_engine config.pbtxt with detected GPU count
cat > /models/dots_ocr_engine/config.pbtxt << EOF
backend: "vllm"
instance_group [
  {
    count: ${NUM_GPUS}
    kind: ${KIND}
  }
]
EOF

echo "[entrypoint] config.pbtxt updated: count=${NUM_GPUS}, kind=${KIND}"

# Start Triton
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
