ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

RUN pip install pymupdf --no-cache-dir

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

# ── Diagnostics ──────────────────────────────────────────────────────────────
tfm_paths = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/transformers.py")
utils_paths = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/utils.py")

if tfm_paths:
    tfm_path = tfm_paths[0]
    lines = open(tfm_path).read().splitlines()
    # Print L685-L710 (load_weights + first mapper)
    print(f"[diag] transformers.py L685-L710:", flush=True)
    for i, line in enumerate(lines[684:710], 685):
        print(f"[diag]   L{i}: {line}", flush=True)

if utils_paths:
    ulines = open(utils_paths[0]).read().splitlines()
    print(f"[diag] utils.py L260-L300:", flush=True)
    for i, line in enumerate(ulines[259:300], 260):
        print(f"[diag]   L{i}: {line}", flush=True)

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

# ── Fix B: register DotsOCRForCausalLM in vLLM model registry ───────────────
# Search in multiple candidate files (registry may not be in __init__.py)
registry_candidates = (
    glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/registry.py") +
    glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/__init__.py")
)

registered = False
for reg_path in registry_candidates:
    src = open(reg_path).read()
    if '"DotsOCRForCausalLM"' in src:
        print(f"[patch-B] already registered in {reg_path}", flush=True)
        registered = True
        break

    # Show lines with Transformers-related entries
    lines = src.splitlines()
    print(f"[patch-B] scanning {reg_path} for insertion point:", flush=True)
    for i, line in enumerate(lines, 1):
        if "Transformers" in line or "_MODELS" in line:
            print(f"[patch-B]   L{i}: {line}", flush=True)

    new_entry = '"DotsOCRForCausalLM": ("vllm.model_executor.models.transformers", "TransformersForCausalLM"),'

    for needle in [
        '"TransformersForMultimodalLM": ("vllm.model_executor.models.transformers", "TransformersForMultimodalLM"),',
        '"TransformersForCausalLM": ("vllm.model_executor.models.transformers", "TransformersForCausalLM"),',
    ]:
        if needle in src:
            for line in lines:
                if needle.strip() in line:
                    indent = " " * (len(line) - len(line.lstrip()))
                    break
            else:
                indent = "    "
            src = src.replace(needle, needle + "\n" + indent + new_entry)
            open(reg_path, "w").write(src)
            print(f"[patch-B] registered DotsOCRForCausalLM in {reg_path}", flush=True)
            registered = True
            break
    if registered:
        break

if not registered:
    print("[patch-B] no suitable registry file found — Fix B skipped", flush=True)

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
