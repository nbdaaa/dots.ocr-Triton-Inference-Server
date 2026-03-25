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
    > /models/dots_mocr/config.pbtxt

echo "[entrypoint] dots_mocr config.pbtxt updated: $NUM_GPUS instance(s), one per GPU"
cat /models/dots_mocr/config.pbtxt

# Scale pipeline CPU instances to match GPU count so pages are forwarded in parallel.
# pipeline is just HTTP forwarding (CPU-bound), safe to run one instance per GPU.
PIPELINE_COUNT=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
cat > /models/pipeline/config.pbtxt << EOF
name: "pipeline"
backend: "python"
max_batch_size: 0

input [
  { name: "PROMPT"     data_type: TYPE_STRING dims: [1] },
  { name: "IMAGE_B64"  data_type: TYPE_STRING dims: [1] },
  { name: "REQUEST_ID" data_type: TYPE_STRING dims: [1] optional: true }
]

output [
  { name: "TEXT" data_type: TYPE_STRING dims: [1] }
]

instance_group [
  { kind: KIND_CPU count: ${PIPELINE_COUNT} }
]

parameters: { key: "engine_model_name" value: { string_value: "dots_mocr" } }
parameters: { key: "max_tokens"        value: { string_value: "24000" } }
parameters: { key: "triton_http_url"   value: { string_value: "http://127.0.0.1:8000" } }
EOF

echo "[entrypoint] pipeline config.pbtxt updated: ${PIPELINE_COUNT} CPU instance(s)"

# Install redis-py for pipeline cancel support
pip install redis --quiet --no-cache-dir

# ── Start Triton OpenAI-compatible frontend ───────────────────────────────────
# Exposes /v1/chat/completions with proper chat template support.
# Runs as a background process; Triton takes over as PID 1 via exec below.
OPENAI_PORT=${OPENAI_FRONTEND_PORT:-9000}
OPENAI_DIR="/opt/tritonserver/python/openai"

if [ -d "${OPENAI_DIR}" ]; then
    if [ -f "${OPENAI_DIR}/requirements.txt" ]; then
        pip install -r "${OPENAI_DIR}/requirements.txt" --quiet --no-cache-dir
    fi
    cd "${OPENAI_DIR}"
    python3 openai_frontend/main.py \
        --model-repository /models \
        --tokenizer rednote-hilab/dots.mocr \
        --port "${OPENAI_PORT}" &
    echo "[entrypoint] OpenAI frontend started on port ${OPENAI_PORT}"
    cd /
else
    echo "[entrypoint] WARNING: OpenAI frontend not found at ${OPENAI_DIR} — chat template will not be applied"
fi

# Start Triton
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
