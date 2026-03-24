#!/bin/bash
set -e

# ── Detect available GPUs ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

# ── Generate pipeline/config.pbtxt ───────────────────────────────────────────
# One GPU instance per physical GPU so each carries its own model replica.
# Falls back to a single CPU instance when no GPU is present (dev/test).
if [ "$NUM_GPUS" -eq 0 ]; then
    INSTANCE_GROUPS="  { kind: KIND_CPU count: 1 }"
else
    INSTANCE_GROUPS=""
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        INSTANCE_GROUPS+="  {\\n    count: 1\\n    kind: KIND_GPU\\n    gpus: [ $i ]\\n  }"
        if [ $i -lt $((NUM_GPUS - 1)) ]; then
            INSTANCE_GROUPS+=",\\n"
        fi
    done
fi

printf "name: \"pipeline\"
backend: \"python\"
max_batch_size: 0

input [
  { name: \"PROMPT\"     data_type: TYPE_STRING dims: [1] },
  { name: \"IMAGE_B64\"  data_type: TYPE_STRING dims: [1] },
  { name: \"REQUEST_ID\" data_type: TYPE_STRING dims: [1] optional: true }
]

output [
  { name: \"TEXT\" data_type: TYPE_STRING dims: [1] }
]

instance_group [
%b
]

parameters: { key: \"model_name\" value: { string_value: \"rednote-hilab/dots.ocr\" } }
parameters: { key: \"max_tokens\"  value: { string_value: \"24000\" } }
" "$INSTANCE_GROUPS" > /models/pipeline/config.pbtxt

echo "[entrypoint] pipeline config.pbtxt updated: ${NUM_GPUS:-0} GPU(s)"
cat /models/pipeline/config.pbtxt

# ── Patch dots.ocr custom module imports ─────────────────────────────────────
# dots.ocr ships trust_remote_code modules that use relative imports which break
# because the dot in "rednote-hilab/dots.ocr" makes Python misparse the package.
# We pre-cache the modules then convert relative → absolute imports.
echo "[fix] Patching dots.ocr module imports..."
python3 - <<'PYEOF'
import sys, os, re

MODULES_DIR = "/root/.cache/huggingface/modules/transformers_modules/rednote-hilab/dots.ocr"

# Step 1: trigger caching if not yet done
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

        # Header: add hash dir to sys.path so bare module names resolve
        header = (
            "# _dots_ocr_fix_applied\n"
            "import sys as _sys, os as _os\n"
            "_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n"
        )
        # from .module import X  →  from module import X
        fixed = re.sub(r"^from \.([\w]+) import", r"from \1 import", content, flags=re.MULTILINE)
        # from . import module  →  import module
        fixed = re.sub(r"^from \. import ([\w]+)", r"import \1", fixed, flags=re.MULTILINE)

        # Footer: register DotsOCRConfig with AutoConfig so trust_remote_code works
        footer = ""
        if "class DotsOCRConfig" in content:
            footer += """
# Register config with transformers AutoConfig
try:
    from transformers import AutoConfig as _AutoConfig
    _AutoConfig.register("dots_ocr", DotsOCRConfig, exist_ok=True)
except Exception:
    pass

# Patch __init__ to inject "AutoModel" into auto_map so AutoModel.from_config
# can load the model class dynamically in the vLLM worker process.
try:
    _orig_dots_init = DotsOCRConfig.__init__
    def _patched_dots_init(self, *_a, **_kw):
        _orig_dots_init(self, *_a, **_kw)
        _am = getattr(self, "auto_map", None)
        if isinstance(_am, dict) and "AutoModelForCausalLM" in _am and "AutoModel" not in _am:
            _am["AutoModel"] = _am["AutoModelForCausalLM"]
    DotsOCRConfig.__init__ = _patched_dots_init
except Exception:
    pass
"""

        with open(fpath, "w") as f:
            f.write(header + fixed + footer)
        print(f"[fix] Patched {fpath}", flush=True)

print("[fix] dots.ocr module patch complete", flush=True)
PYEOF

# ── Runtime dependencies ──────────────────────────────────────────────────────
pip install redis qwen-vl-utils --quiet --no-cache-dir

# ── Start Triton ──────────────────────────────────────────────────────────────
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
