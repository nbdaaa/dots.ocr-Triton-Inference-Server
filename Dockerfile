ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

RUN pip install pymupdf --no-cache-dir

# Patch vLLM model registry to map DotsOCRForCausalLM → TransformersForMultimodalLM.
# Without this, vLLM picks TransformersForCausalLM (based on "ForCausalLM" suffix),
# which cannot load the vision_tower weights that live inside the multimodal model.
RUN python3 - <<'EOF'
import sys, glob, re

# Locate the vLLM models __init__.py that contains _MODELS / ModelRegistry
candidates = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/__init__.py")
if not candidates:
    print("[patch] vLLM models __init__.py not found — skipping", flush=True)
    sys.exit(0)

path = candidates[0]
with open(path) as f:
    src = f.read()

entry = '"DotsOCRForCausalLM"'
if entry in src:
    print("[patch] DotsOCRForCausalLM already registered — skipping", flush=True)
    sys.exit(0)

# Insert alongside other Transformers-based entries.
# The registry uses a dict literal like:
#   "TransformersForMultimodalLM": ("vllm.model_executor.models.transformers", "TransformersForMultimodalLM"),
# We append our entry right after that line.
needle = '"TransformersForMultimodalLM": ("vllm.model_executor.models.transformers", "TransformersForMultimodalLM"),'
replacement = (
    needle + "\n"
    '    "DotsOCRForCausalLM": ("vllm.model_executor.models.transformers", "TransformersForMultimodalLM"),  # dots.ocr patch'
)

if needle not in src:
    # Fallback: try without trailing comma
    needle2 = '"TransformersForMultimodalLM": ("vllm.model_executor.models.transformers", "TransformersForMultimodalLM")'
    if needle2 not in src:
        print(f"[patch] Anchor not found in {path} — skipping", flush=True)
        sys.exit(0)
    replacement = (
        needle2 + ",\n"
        '    "DotsOCRForCausalLM": ("vllm.model_executor.models.transformers", "TransformersForMultimodalLM"),  # dots.ocr patch'
    )
    src = src.replace(needle2, replacement, 1)
else:
    src = src.replace(needle, replacement, 1)

with open(path, "w") as f:
    f.write(src)
print(f"[patch] DotsOCRForCausalLM → TransformersForMultimodalLM registered in {path}", flush=True)
EOF

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
