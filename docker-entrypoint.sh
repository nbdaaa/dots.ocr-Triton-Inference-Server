#!/bin/bash
set -e

# ── Detect available GPUs ─────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

# ── Compute max batch size from GPU memory ────────────────────────────────────
# Uses the first GPU's total memory as the reference.
# Rough budget:
#   - Model weights (7B bfloat16)  ≈ 14 000 MB
#   - Per-request overhead
#     (KV cache + activations for a typical OCR page) ≈ 2 000 MB
#   - Safety headroom                                 ≈ 1 000 MB
# Capped at 8 to avoid excessive queue latency.
MODEL_MEM_MB=8000    # actual: ~7500 MB for 7B bfloat16
PER_REQ_MEM_MB=2000  # KV cache + activations per OCR page (up to 24K tokens)
SAFETY_MB=2000       # headroom for image pixel tensors and fragmentation

if [ "$NUM_GPUS" -gt 0 ]; then
    TOTAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1)
    MAX_BATCH=$(( (TOTAL_GPU_MEM - MODEL_MEM_MB - SAFETY_MB) / PER_REQ_MEM_MB ))
    [ "$MAX_BATCH" -lt 1 ] && MAX_BATCH=1
    [ "$MAX_BATCH" -gt 8 ] && MAX_BATCH=8
else
    MAX_BATCH=1
fi

echo "[entrypoint] GPU total mem: ${TOTAL_GPU_MEM:-0} MB → max_batch_size=${MAX_BATCH}"

# Build preferred_batch_size list (powers of 2 up to MAX_BATCH)
PREFERRED_SIZES=""
B=1
while [ "$B" -le "$MAX_BATCH" ]; do
    [ -n "$PREFERRED_SIZES" ] && PREFERRED_SIZES+=", "
    PREFERRED_SIZES+="$B"
    B=$(( B * 2 ))
done

# ── Generate pipeline/config.pbtxt ───────────────────────────────────────────
# One GPU instance per physical GPU; each instance loads its own model replica
# and handles requests independently (Triton load-balances across instances).
# dynamic_batching lets Triton accumulate up to MAX_BATCH pending requests and
# send them together in a single execute() call — the model.py runs them as
# one batched model.generate() for higher GPU utilisation.
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
max_batch_size: %d

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

dynamic_batching {
  preferred_batch_size: [ %s ]
  max_queue_delay_microseconds: 500000
}

parameters: { key: \"model_name\" value: { string_value: \"rednote-hilab/dots.mocr\" } }
parameters: { key: \"max_tokens\"  value: { string_value: \"24000\" } }
" "$MAX_BATCH" "$INSTANCE_GROUPS" "$PREFERRED_SIZES" > /models/pipeline/config.pbtxt

echo "[entrypoint] pipeline config.pbtxt updated: ${NUM_GPUS:-0} GPU(s), max_batch_size=${MAX_BATCH}"
cat /models/pipeline/config.pbtxt

# ── Patch dots.mocr custom module imports ─────────────────────────────────────
echo "[fix] Patching dots.mocr module imports..."
python3 - <<'PYEOF'
import sys, os, re

MODULES_DIR = "/root/.cache/huggingface/modules/transformers_modules/rednote-hilab/dots.mocr"

if not os.path.isdir(MODULES_DIR):
    print("[fix] Pre-caching dots.mocr custom modules...", flush=True)
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("rednote-hilab/dots.mocr", trust_remote_code=True)
    except Exception as e:
        print(f"[fix] Pre-cache done (import error expected): {type(e).__name__}", flush=True)

if not os.path.isdir(MODULES_DIR):
    print("[fix] Modules dir not found after pre-cache — skipping patch", flush=True)
    sys.exit(0)

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
        if "_dots_mocr_fix_applied" in content:
            continue

        header = (
            "# _dots_mocr_fix_applied\n"
            "import sys as _sys, os as _os\n"
            "_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n"
        )
        fixed = re.sub(r"^from \.([\w]+) import", r"from \1 import", content, flags=re.MULTILINE)
        fixed = re.sub(r"^from \. import ([\w]+)", r"import \1", fixed, flags=re.MULTILINE)

        footer = ""
        if "class DotsMOCRConfig" in content:
            footer += """
try:
    from transformers import AutoConfig as _AutoConfig
    _AutoConfig.register("dots_mocr", DotsMOCRConfig, exist_ok=True)
except Exception:
    pass

try:
    _orig_dots_init = DotsMOCRConfig.__init__
    def _patched_dots_init(self, *_a, **_kw):
        _orig_dots_init(self, *_a, **_kw)
        _am = getattr(self, "auto_map", None)
        if isinstance(_am, dict) and "AutoModelForCausalLM" in _am and "AutoModel" not in _am:
            _am["AutoModel"] = _am["AutoModelForCausalLM"]
    DotsMOCRConfig.__init__ = _patched_dots_init
except Exception:
    pass
"""

        with open(fpath, "w") as f:
            f.write(header + fixed + footer)
        print(f"[fix] Patched {fpath}", flush=True)

print("[fix] dots.mocr module patch complete", flush=True)
PYEOF

# ── Runtime dependencies ──────────────────────────────────────────────────────
pip install redis qwen-vl-utils accelerate --quiet --no-cache-dir

# ── Start Triton ──────────────────────────────────────────────────────────────
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
