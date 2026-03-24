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

parameters: { key: "engine_model_name" value: { string_value: "dots_ocr" } }
parameters: { key: "max_tokens"        value: { string_value: "24000" } }
parameters: { key: "triton_http_url"   value: { string_value: "http://127.0.0.1:8000" } }
EOF

echo "[entrypoint] pipeline config.pbtxt updated: ${PIPELINE_COUNT} CPU instance(s)"

# Fix: dots.ocr custom modules use relative imports that break because the dot
# in "rednote-hilab/dots.ocr" makes Python misparse the module package path
# (it looks for a "dots" package instead of "dots.ocr" directory).
# Solution: pre-cache the modules then patch relative → absolute imports.
echo "[fix] Patching dots.ocr module imports..."
python3 - <<'PYEOF'
import sys, os, re

MODULES_DIR = "/root/.cache/huggingface/modules/transformers_modules/rednote-hilab/dots.ocr"

# Step 1: if modules not yet cached, trigger caching via AutoConfig
# (the import will fail but the .py files still get written to MODULES_DIR)
if not os.path.isdir(MODULES_DIR):
    print("[fix] Pre-caching dots.ocr custom modules...", flush=True)
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("rednote-hilab/dots.ocr", trust_remote_code=True)
    except Exception as e:
        print(f"[fix] Pre-cache done (import error expected): {type(e).__name__}", flush=True)

if not os.path.isdir(MODULES_DIR):
    print("[fix] Modules dir not found after pre-cache — skipping patch", flush=True)
    sys.exit(0)

# Step 2: patch every .py file in every hash sub-directory
for hash_dir in os.listdir(MODULES_DIR):
    hash_path = os.path.join(MODULES_DIR, hash_dir)
    if not os.path.isdir(hash_path):
        continue
    for fname in os.listdir(hash_path):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(hash_path, fname)
        with open(fpath) as f:
            content = f.read()
        if "_dots_ocr_fix_applied" in content:
            continue

        # Header: add hash dir to sys.path so bare module names resolve correctly
        header = (
            "# _dots_ocr_fix_applied\n"
            "import sys as _sys, os as _os\n"
            "_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n"
        )
        # from .module import X  →  from module import X
        fixed = re.sub(r"^from \.([\w]+) import", r"from \1 import", content, flags=re.MULTILINE)
        # from . import module  →  import module
        fixed = re.sub(r"^from \. import ([\w]+)", r"import \1", fixed, flags=re.MULTILINE)

        # Footer: auto-register classes with AutoConfig/AutoModel so that
        # TransformersForMultimodalLM's AutoModel.from_config() can find them
        # in the vLLM worker process when these files are imported.
        footer = ""
        if "class DotsOCRConfig" in content:
            footer += """
# Register config with transformers AutoConfig so AutoModel.from_config works
try:
    from transformers import AutoConfig as _AutoConfig
    _AutoConfig.register("dots_ocr", DotsOCRConfig, exist_ok=True)
except Exception:
    pass
"""
        if "class DotsOCRForCausalLM" in content:
            footer += """
# Register model with transformers AutoModel
try:
    from transformers import AutoModel as _AutoModel
    from configuration_dots import DotsOCRConfig as _DotsOCRConfig
    _AutoModel.register(_DotsOCRConfig, DotsOCRForCausalLM, exist_ok=True)
except Exception:
    pass
"""

        with open(fpath, "w") as f:
            f.write(header + fixed + footer)
        print(f"[fix] Patched {fpath}", flush=True)

print("[fix] dots.ocr module patch complete", flush=True)
PYEOF

# Install redis-py for pipeline cancel support
pip install redis --quiet --no-cache-dir

# Start Triton
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
