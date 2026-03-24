ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

RUN pip install pymupdf --no-cache-dir

# Allow None for optional multi-modal sub-processors (e.g. video_processor in
# Qwen2_5_VLProcessor).  dots.ocr is image-only, so its DotsOCRProcessor never
# passes video_processor, which defaults to None and then fails the strict type
# check added in newer transformers.  Inserting an early return for None is safe:
# the attribute is simply not registered on the processor instance.
# Uses regex to match the method signature regardless of type annotations.
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

# Match method def with any signature (with or without type annotations)
m = re.search(r'def check_argument_for_proper_class\(self[^)]*\)[^:]*:', src)
if not m:
    print("[patch] check_argument_for_proper_class not found — skipping", flush=True)
    sys.exit(0)

sig_text = src[m.start():m.end()]
print(f"[patch] found signature: {sig_text!r}", flush=True)

# Extract actual parameter names from signature (strip type annotations)
# e.g. "(self, attribute_name: str, value: Any) -> None:" → ['self', 'attribute_name', 'value']
paren_m = re.search(r'\((.+)\)', sig_text, re.DOTALL)
if not paren_m:
    print("[patch] could not parse parameter list — skipping", flush=True)
    sys.exit(0)
param_names = [re.match(r'\s*(\w+)', p).group(1)
               for p in paren_m.group(1).split(',')
               if re.match(r'\s*\w+', p)]
print(f"[patch] parameter names: {param_names}", flush=True)
# The second non-self parameter is the value being type-checked
arg_param = param_names[2] if len(param_names) > 2 else 'arg'

body_start = src.index("\n", m.end()) + 1
j = body_start
while j < len(src) and src[j] in (' ', '\t'):
    j += 1
indent = src[body_start:j]
guard = (
    f"{indent}if {arg_param} is None:  # _dots_ocr_allow_none: skip check for absent optional processors\n"
    f"{indent}    return\n"
)
patched = src[:body_start] + guard + src[body_start:]
with open(path, "w") as f:
    f.write(patched)
print(f"[patch] {path} patched — None allowed for optional processors (param: {arg_param!r})", flush=True)
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
                f"{indent}# _dots_ocr_remap: fix weight key paths for dots.ocr (Qwen2.5-VL based).\n"
                f"{indent}# TransformersForMultimodalLM mapper (designed for InternVL) may convert\n"
                f"{indent}# model.model.X → model.language_model.X, but Qwen2.5-VL stores the LM\n"
                f"{indent}# as self.model (not self.language_model).  Also handles raw keys that\n"
                f"{indent}# need 'model.' prepended when no mapper is active.\n"
                f"{indent}try:\n"
                f"{indent}    _vllm_top = set(dict(self.module.named_children()).keys())\n"
                f"{indent}    _hf_children = (set(dict(self.module.model.named_children()).keys())\n"
                f"{indent}                    if hasattr(self.module, 'model') and\n"
                f"{indent}                       hasattr(self.module.model, 'named_children') else set())\n"
                f"{indent}    def _dots_ocr_fix_key(_name):\n"
                f"{indent}        _parts = _name.split('.')\n"
                f"{indent}        if not _parts:\n"
                f"{indent}            return _name\n"
                f"{indent}        # Case 1: raw key without 'model.' prefix (e.g. TransformersForCausalLM\n"
                f"{indent}        # with no mapper): prepend 'model.' when key is a known HF child.\n"
                f"{indent}        if _parts[0] not in _vllm_top and _parts[0] in _hf_children:\n"
                f"{indent}            return 'model.' + _name\n"
                f"{indent}        # Case 2: mapper created model.language_model.X but HF model stores\n"
                f"{indent}        # it as model.model.X (Qwen2.5-VL: self.model = language model).\n"
                f"{indent}        if (_parts[0] == 'model' and len(_parts) >= 2 and\n"
                f"{indent}                _parts[1] == 'language_model' and\n"
                f"{indent}                'language_model' not in _hf_children and\n"
                f"{indent}                'model' in _hf_children):\n"
                f"{indent}            return 'model.model.' + '.'.join(_parts[2:])\n"
                f"{indent}        # Case 3: mapper created model.vision_tower.X but HF model uses visual.\n"
                f"{indent}        if (_parts[0] == 'model' and len(_parts) >= 2 and\n"
                f"{indent}                _parts[1] == 'vision_tower' and\n"
                f"{indent}                'vision_tower' not in _hf_children and\n"
                f"{indent}                'visual' in _hf_children):\n"
                f"{indent}            return 'model.visual.' + '.'.join(_parts[2:])\n"
                f"{indent}        return _name\n"
                f"{indent}    # Case 4: synthesise lm_head.weight from embed_tokens.weight when the\n"
                f"{indent}    # checkpoint omits it (tie_word_embeddings=True).  We track whether the\n"
                f"{indent}    # real lm_head.weight was seen in the stream to avoid overwriting it.\n"
                f"{indent}    import itertools as _itertools\n"
                f"{indent}    _seen_lm_head = set()\n"
                f"{indent}    def _dots_ocr_gen(_n, _t):\n"
                f"{indent}        _fixed = _dots_ocr_fix_key(_n)\n"
                f"{indent}        if 'lm_head.weight' in _fixed:\n"
                f"{indent}            _seen_lm_head.add(_fixed)\n"
                f"{indent}        yield (_fixed, _t)\n"
                f"{indent}        if 'embed_tokens.weight' in _fixed:\n"
                f"{indent}            _lm = _fixed.replace('embed_tokens.weight', 'lm_head.weight')\n"
                f"{indent}            if _lm != _fixed and _lm not in _seen_lm_head:\n"
                f"{indent}                yield (_lm, _t)\n"
                f"{indent}    weights = _itertools.chain.from_iterable(\n"
                f"{indent}        _dots_ocr_gen(_n, _t) for _n, _t in weights)\n"
                f"{indent}except Exception as _e:\n"
                f"{indent}    import traceback as _tb; _tb.print_exc()\n"
                f"{indent}\n"
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

# Patch vLLM default_loader to handle tied embeddings (lm_head.weight tied to
# embed_tokens.weight when tie_word_embeddings=True).  The checkpoint omits the
# tied weight; vLLM's post-load validator raises ValueError seeing it unloaded.
# Fix: call tie_weights() on the HF model then mark tied params as loaded.
RUN python3 - <<'EOF'
import glob

paths = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/model_loader/default_loader.py")
if not paths:
    print("[patch-tie] default_loader.py not found — skipping", flush=True)
    exit(0)

path = paths[0]
src = open(path).read()

if "_dots_ocr_tie_weights" in src:
    print("[patch-tie] already patched", flush=True)
    exit(0)

needle = 'raise ValueError("Following weights were not initialized from '
if needle not in src:
    print(f"[patch-tie] needle not found — skipping", flush=True)
    exit(0)

idx = src.index(needle)
line_start = src.rfind("\n", 0, idx) + 1
indent = " " * (idx - line_start)

patch = (
    f"{indent}# _dots_ocr_tie_weights: handle lm_head.weight tied to embed_tokens.weight\n"
    f"{indent}# (tie_word_embeddings=True → checkpoint omits lm_head.weight).\n"
    f"{indent}# 1) Call tie_weights() on the HF model so lm_head shares embed_tokens storage.\n"
    f"{indent}# 2) Detect tied params (same data_ptr as a loaded param) and mark loaded.\n"
    f"{indent}try:\n"
    f"{indent}    import sys as _sys\n"
    f"{indent}    # Step 1: tie the weights in the HF model\n"
    f"{indent}    _hf = getattr(model, 'model', None)\n"
    f"{indent}    if _hf is not None and hasattr(_hf, 'tie_weights'):\n"
    f"{indent}        _hf.tie_weights()\n"
    f"{indent}    # Step 2: rebuild param dict and find tied params\n"
    f"{indent}    _params = dict(model.named_parameters())\n"
    f"{indent}    _loaded_ptrs = {{}}\n"
    f"{indent}    for _n in list(loaded_weights):\n"
    f"{indent}        _p = _params.get(_n)\n"
    f"{indent}        if _p is not None:\n"
    f"{indent}            _loaded_ptrs[_p.data_ptr()] = _n\n"
    f"{indent}    _newly_tied = []\n"
    f"{indent}    for _n, _p in _params.items():\n"
    f"{indent}        if _n not in loaded_weights and _p.data_ptr() in _loaded_ptrs:\n"
    f"{indent}            _newly_tied.append(_n)\n"
    f"{indent}    for _n in _newly_tied:\n"
    f"{indent}        loaded_weights.add(_n)\n"
    f"{indent}    if _newly_tied:\n"
    f"{indent}        print(f'[patch-tie] resolved tied weights: {{_newly_tied}}', file=_sys.stderr, flush=True)\n"
    f"{indent}except Exception as _e:\n"
    f"{indent}    import sys as _sys, traceback as _tb\n"
    f"{indent}    print(f'[patch-tie FAILED] {{type(_e).__name__}}: {{_e}}', file=_sys.stderr, flush=True)\n"
    f"{indent}    _tb.print_exc(file=_sys.stderr)\n"
    f"{indent}"
)

patched = src[:idx] + patch + src[idx:]
open(path, "w").write(patched)
print(f"[patch-tie] {path} patched — tied weights handled before uninitialized check", flush=True)
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
