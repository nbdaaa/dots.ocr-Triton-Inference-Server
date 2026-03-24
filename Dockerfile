ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

RUN pip install pymupdf --no-cache-dir

# Allow None for optional multi-modal sub-processors (e.g. video_processor in
# Qwen2_5_VLProcessor).  dots.ocr is image-only, so its DotsOCRProcessor never
# passes video_processor, which defaults to None and then fails the strict type
# check added in newer transformers.  Inserting an early return for None is safe:
# the attribute is simply not registered on the processor instance.
RUN python3 - <<'EOF'
import glob, sys

paths = glob.glob("/usr/local/lib/python3*/dist-packages/transformers/processing_utils.py")
if not paths:
    print("[patch] transformers/processing_utils.py not found — skipping", flush=True)
    sys.exit(0)

path = paths[0]
with open(path) as f:
    src = f.read()

marker = "def check_argument_for_proper_class(self, attribute_name, arg):"
if "_dots_ocr_allow_none" in src:
    print("[patch] processing_utils.py already patched — skipping", flush=True)
    sys.exit(0)
if marker not in src:
    print(f"[patch] {marker!r} not found — skipping", flush=True)
    sys.exit(0)

idx = src.index(marker)
body_start = src.index("\n", idx) + 1
j = body_start
while j < len(src) and src[j] in (' ', '\t'):
    j += 1
indent = src[body_start:j]
guard = (
    f"{indent}if arg is None:  # _dots_ocr_allow_none: skip check for absent optional processors\n"
    f"{indent}    return\n"
)
patched = src[:body_start] + guard + src[body_start:]
with open(path, "w") as f:
    f.write(patched)
print(f"[patch] {path} patched — None is now allowed for optional processors", flush=True)
EOF

# Patch vLLM to load DotsOCR (a multimodal model) correctly.
#
# Root cause: DotsOCRForCausalLM is not in vLLM's registry, so vLLM falls back
# to trust_remote_code and picks TransformersForCausalLM whose hf_to_vllm_mapper
# only remaps "model." → "model.model." but leaves "vision_tower.*" keys unmapped.
# The weight loader then can't find vision_tower in the vLLM wrapper → ValueError.
#
# Fix A: patch transformers.py so that TransformersForCausalLM's mapper prepends
#        "model." to ALL checkpoint keys (handles vision_tower.* correctly).
# Fix B: register DotsOCRForCausalLM in the vLLM model registry pointing to
#        TransformersForMultimodalLM, which already has the correct mapper.
RUN python3 - <<'EOF'
import sys, glob, re

utils_paths = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/utils.py")

# ── Fix A: patch load_weights in utils.py to remap unrecognised top-level keys ─
# The mapper in TransformersForCausalLM has "vision_tower"→"model.vision_tower",
# but if WeightsMapper prefix-matching requires an exact separator match and the
# key is e.g. "vision_tower.encoder.weight" the prefix lookup may still fail.
# Safest fix: BEFORE _load_module is called, programmatically remap any weight
# whose first path component is not a named child of the vLLM wrapper but IS a
# named child of wrapper.model (the HF model).
if utils_paths:
    utils_path = utils_paths[0]
    src = open(utils_path).read()

    if "_dots_ocr_remap" in src:
        print("[patch-A] utils.py remap already applied", flush=True)
    else:
        # Find the line that calls _load_module from load_weights and insert
        # the remap block immediately before it.
        needle = "autoloaded_weights = set(self._load_module("
        if needle in src:
            # Detect indentation of that line
            idx = src.index(needle)
            line_start = src.rfind("\n", 0, idx) + 1
            indent = " " * (idx - line_start)
            remap_block = (
                f"{indent}# _dots_ocr_remap: prepend 'model.' to top-level weight keys that\n"
                f"{indent}# the mapper missed but that exist under self.module.model\n"
                f"{indent}try:\n"
                f"{indent}    _vllm_top = set(dict(self.module.named_children()).keys())\n"
                f"{indent}    _hf_top = set(dict(self.module.model.named_children()).keys()) \\\n"
                f"{indent}              if hasattr(self.module, 'model') and \\\n"
                f"{indent}                 hasattr(self.module.model, 'named_children') else set()\n"
                f"{indent}    _weights_list = list(weights)\n"
                f"{indent}    weights = [\n"
                f"{indent}        ('model.' + _n, _t)\n"
                f"{indent}        if _n.split('.')[0] not in _vllm_top and _n.split('.')[0] in _hf_top\n"
                f"{indent}        else (_n, _t)\n"
                f"{indent}        for _n, _t in _weights_list\n"
                f"{indent}    ]\n"
                f"{indent}except Exception:\n"
                f"{indent}    pass\n"
                f"{indent}"
            )
            patched = src[:idx] + remap_block + src[idx:]
            open(utils_path, "w").write(patched)
            print("[patch-A] utils.py remap block inserted before _load_module call", flush=True)
        else:
            print("[patch-A] _load_module call not found in utils.py — skipping", flush=True)

# ── Fix B: register DotsOCRForCausalLM → TransformersForMultimodalLM ─────────
# registry.py uses short names like ("transformers", "TransformersForMultimodalLM")
# The target dict is _TRANSFORMERS_SUPPORTED_MODELS.
# Anchor: "Emu3ForConditionalGeneration": ("transformers", "TransformersForMultimodalLM")
reg_paths = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/registry.py")
if not reg_paths:
    print("[patch-B] registry.py not found — skipping", flush=True)
else:
    reg_path = reg_paths[0]
    src = open(reg_path).read()

    if '"DotsOCRForCausalLM"' in src:
        print("[patch-B] DotsOCRForCausalLM already registered", flush=True)
    else:
        new_line = '    "DotsOCRForCausalLM": ("transformers", "TransformersForMultimodalLM"),  # dots.ocr patch'
        inserted = False

        # Strategy 1: insert after the Emu3 entry (confirmed to exist in the file)
        for anchor in [
            '"Emu3ForConditionalGeneration": ("transformers", "TransformersForMultimodalLM"),  # noqa: E501',
            '"Emu3ForConditionalGeneration": ("transformers", "TransformersForMultimodalLM"),',
            '"SmolLM3ForCausalLM": ("transformers", "TransformersForCausalLM"),',
        ]:
            if anchor in src:
                src = src.replace(anchor, anchor + "\n" + new_line)
                open(reg_path, "w").write(src)
                print(f"[patch-B] DotsOCRForCausalLM registered in registry.py (anchor: {anchor[:40]}...)", flush=True)
                inserted = True
                break

        if not inserted:
            # Strategy 2: insert at end of _TRANSFORMERS_SUPPORTED_MODELS dict
            m = re.search(r'(_TRANSFORMERS_SUPPORTED_MODELS\s*=\s*\{[^}]*)\}', src, re.DOTALL)
            if m:
                src = src[:m.end()-1] + new_line + "\n}" + src[m.end():]
                open(reg_path, "w").write(src)
                print("[patch-B] DotsOCRForCausalLM registered (end of _TRANSFORMERS_SUPPORTED_MODELS)", flush=True)
                inserted = True

        if not inserted:
            print("[patch-B] no anchor found — Fix B skipped", flush=True)

print("[patch] Done", flush=True)
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
