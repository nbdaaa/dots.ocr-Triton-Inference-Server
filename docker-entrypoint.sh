#!/bin/bash
set -e

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

# Build one instance_group block per GPU with explicit GPU assignment.
# This gives Triton a dedicated vLLM instance per GPU (Option B).
# tensor_parallel_size=1 in model.json ensures each instance uses only its assigned GPU.
if [ "$NUM_GPUS" -eq 0 ]; then
    INSTANCE_GROUPS="  {\n    count: 1\n    kind: KIND_CPU\n  }"
else
    INSTANCE_GROUPS=""
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        INSTANCE_GROUPS+="  {\n    count: 1\n    kind: KIND_GPU\n    gpus: [ $i ]\n  }"
        if [ $i -lt $((NUM_GPUS - 1)) ]; then
            INSTANCE_GROUPS+=",\n"
        fi
    done
fi

printf "backend: \"vllm\"\ninstance_group [\n%b\n]\n" "$INSTANCE_GROUPS" \
    > /models/dots_ocr/config.pbtxt

echo "[entrypoint] config.pbtxt updated: $NUM_GPUS instance(s), one per GPU"
cat /models/dots_ocr/config.pbtxt

# Start Triton
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
