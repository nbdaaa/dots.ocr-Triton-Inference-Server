ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}
# Recommended tag for PyTorch backend (no vLLM overhead):
#   TRITON_IMAGE_TAG=25.10-pyt-python-py3
# The vllm-python-py3 variant also works since it includes transformers.

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    pillow \
    pymupdf \
    redis \
    qwen-vl-utils

# ── Patch transformers: allow None for optional multi-modal sub-processors ────
# DotsOCRProcessor (Qwen2.5-VL based) never passes video_processor (image-only
# model).  ProcessorMixin.check_argument_for_proper_class rejects None values
# for registered attributes; inserting an early-return for None is safe.
RUN python3 - <<'EOF'
import glob, re, sys

paths = glob.glob("/usr/local/lib/python3*/dist-packages/transformers/processing_utils.py")
if not paths:
    print("[patch] transformers/processing_utils.py not found — skipping", flush=True)
    sys.exit(0)

path = paths[0]
with open(path) as f:
    src = f.read()

if "_dots_ocr_allow_none" in src:
    print("[patch] processing_utils.py already patched — skipping", flush=True)
    sys.exit(0)

m = re.search(r'def check_argument_for_proper_class\(self[^)]*\)[^:]*:', src)
if not m:
    print("[patch] check_argument_for_proper_class not found — skipping", flush=True)
    sys.exit(0)

sig_text = src[m.start():m.end()]
paren_m  = re.search(r'\((.+)\)', sig_text, re.DOTALL)
if not paren_m:
    print("[patch] could not parse parameter list — skipping", flush=True)
    sys.exit(0)

param_names = [re.match(r'\s*(\w+)', p).group(1)
               for p in paren_m.group(1).split(',')
               if re.match(r'\s*\w+', p)]
arg_param = param_names[2] if len(param_names) > 2 else 'arg'

body_start = src.index("\n", m.end()) + 1
j = body_start
while j < len(src) and src[j] in (' ', '\t'):
    j += 1
indent = src[body_start:j]
guard = (
    f"{indent}if {arg_param} is None:  # _dots_ocr_allow_none\n"
    f"{indent}    return\n"
)
patched = src[:body_start] + guard + src[body_start:]
with open(path, "w") as f:
    f.write(patched)
print(f"[patch] {path} patched (param: {arg_param!r})", flush=True)
EOF
