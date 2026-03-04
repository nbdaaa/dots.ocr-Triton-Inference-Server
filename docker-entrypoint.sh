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

echo "[entrypoint] dots_ocr config.pbtxt updated: $NUM_GPUS instance(s), one per GPU"
cat /models/dots_ocr/config.pbtxt

# Scale pipeline CPU instances to match GPU count so pages are forwarded in parallel.
# pipeline is just HTTP forwarding (CPU-bound), safe to run one instance per GPU.
PIPELINE_COUNT=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
cat > /models/pipeline/config.pbtxt << EOF
name: "pipeline"
backend: "python"
max_batch_size: 0

input [
  { name: "PROMPT"    data_type: TYPE_STRING dims: [1] },
  { name: "IMAGE_B64" data_type: TYPE_STRING dims: [1] }
]

output [
  { name: "TEXT" data_type: TYPE_STRING dims: [1] }
]

instance_group [
  { kind: KIND_CPU count: ${PIPELINE_COUNT} }
]

parameters: { key: "engine_model_name" value: { string_value: "dots_ocr" } }
parameters: { key: "max_tokens"        value: { string_value: "24000" } }
parameters: { key: "triton_http_url"   value: { string_value: "http://127.0.0.1:8000" } }
EOF

echo "[entrypoint] pipeline config.pbtxt updated: ${PIPELINE_COUNT} CPU instance(s)"

# Start Triton
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
