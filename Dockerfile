ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

RUN pip install pymupdf --no-cache-dir

# Patch vLLM backend to respect Triton's GPU assignment via CUDA_VISIBLE_DEVICES.
# Without this, vLLM ignores instance_group gpus: [N] and always uses GPU 0.
# Fix: https://github.com/triton-inference-server/server/issues/6855
RUN python3 - <<'EOF'
import re, sys

path = "/opt/tritonserver/backends/vllm/model.py"
try:
    with open(path) as f:
        src = f.read()
except FileNotFoundError:
    print(f"[patch] {path} not found — skipping", flush=True)
    sys.exit(0)

patch = (
    '    import os\n'
    '    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.get("model_instance_device_id", "0"))\n'
)

marker = "async def initialize(self, args):"
if patch.strip() in src:
    print("[patch] already applied — skipping", flush=True)
    sys.exit(0)
if marker not in src:
    print(f"[patch] marker '{marker}' not found — skipping", flush=True)
    sys.exit(0)

patched = src.replace(marker, marker + "\n" + patch, 1)
with open(path, "w") as f:
    f.write(patched)
print("[patch] CUDA_VISIBLE_DEVICES patch applied to vLLM backend", flush=True)
EOF
