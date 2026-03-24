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
import sys, glob, re, textwrap

# ── Fix A: patch hf_to_vllm_mapper in transformers.py ───────────────────────
tfm_paths = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/transformers.py")
if not tfm_paths:
    print("[patch-A] transformers.py not found — skipping Fix A", flush=True)
else:
    tfm_path = tfm_paths[0]
    with open(tfm_path) as f:
        src = f.read()

    # Print existing mapper lines for diagnostics
    print(f"[patch-A] Examining {tfm_path}", flush=True)
    for i, line in enumerate(src.splitlines(), 1):
        if "hf_to_vllm_mapper" in line or ("WeightsMapper" in line and "model." in line):
            print(f"[patch-A]   L{i}: {line}", flush=True)

    if "_dots_ocr_mapper_fix" in src:
        print("[patch-A] mapper fix already applied — skipping", flush=True)
    else:
        # Strategy: find the hf_to_vllm_mapper property inside TransformersForCausalLM
        # and replace any WeightsMapper that only maps "model." → "model.model."
        # with one that maps "" → "model." (prepend model. to ALL keys).
        #
        # Pattern: orig_to_new_prefix={"model.": "model.model."}
        # Replace with: orig_to_new_prefix={"": "model."}  # _dots_ocr_mapper_fix
        old_mapper = re.search(
            r'orig_to_new_prefix\s*=\s*\{\s*"model\."\s*:\s*"model\.model\."\s*\}',
            src
        )
        if old_mapper:
            patched = src[:old_mapper.start()] + \
                      'orig_to_new_prefix={"": "model."}  # _dots_ocr_mapper_fix' + \
                      src[old_mapper.end():]
            with open(tfm_path, "w") as f:
                f.write(patched)
            print(f'[patch-A] hf_to_vllm_mapper fixed: "model."→"model.model." replaced with ""→"model."', flush=True)
        else:
            # Mapper pattern not found as expected — print context for diagnostics
            print("[patch-A] expected mapper pattern not found; dumping hf_to_vllm_mapper context:", flush=True)
            in_class = False
            in_mapper = False
            for i, line in enumerate(src.splitlines(), 1):
                if "class TransformersForCausalLM" in line:
                    in_class = True
                if in_class and "hf_to_vllm_mapper" in line:
                    in_mapper = True
                if in_mapper:
                    print(f"[patch-A]   L{i}: {line}", flush=True)
                    if i > 0 and "return" in line:
                        in_mapper = False
                        break

# ── Fix B: register DotsOCRForCausalLM in vLLM model registry ───────────────
init_paths = glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/models/__init__.py")
if not init_paths:
    print("[patch-B] models/__init__.py not found — skipping Fix B", flush=True)
else:
    init_path = init_paths[0]
    with open(init_path) as f:
        src = f.read()

    if '"DotsOCRForCausalLM"' in src:
        print("[patch-B] DotsOCRForCausalLM already registered", flush=True)
    else:
        # Print first 80 lines for diagnostics so we can see the actual format
        lines = src.splitlines()
        print(f"[patch-B] Registry format preview ({init_path}):", flush=True)
        for i, line in enumerate(lines[:80], 1):
            if "Transformers" in line or "_MODELS" in line or "register" in line.lower():
                print(f"[patch-B]   L{i}: {line}", flush=True)

        # Try multiple insertion strategies
        inserted = False

        # Strategy 1: dict literal with TransformersForMultimodalLM entry
        for needle in [
            '"TransformersForMultimodalLM": ("vllm.model_executor.models.transformers", "TransformersForMultimodalLM"),',
            '"TransformersForCausalLM": ("vllm.model_executor.models.transformers", "TransformersForCausalLM"),',
        ]:
            if needle in src:
                new_entry = (
                    '"DotsOCRForCausalLM": ("vllm.model_executor.models.transformers", '
                    '"TransformersForMultimodalLM"),  # dots.ocr patch'
                )
                # Match indentation of the needle line
                for line in lines:
                    if needle.strip() in line:
                        indent = len(line) - len(line.lstrip())
                        break
                else:
                    indent = 4
                src = src.replace(needle, needle + "\n" + " " * indent + new_entry)
                with open(init_path, "w") as f:
                    f.write(src)
                print(f"[patch-B] DotsOCRForCausalLM registered (strategy 1)", flush=True)
                inserted = True
                break

        if not inserted:
            # Strategy 2: ModelRegistry.register() call style
            reg_match = re.search(r'ModelRegistry\s*\.\s*register\s*\(', src)
            if reg_match:
                # Append a register call at module level after imports
                append = textwrap.dedent("""
                # dots.ocr patch
                try:
                    from vllm.model_executor.models.transformers import TransformersForMultimodalLM as _TMMLM
                    ModelRegistry.register_model("DotsOCRForCausalLM", _TMMLM)
                except Exception as _e:
                    pass
                """)
                with open(init_path, "a") as f:
                    f.write(append)
                print("[patch-B] DotsOCRForCausalLM registered (strategy 2 - append)", flush=True)
                inserted = True

        if not inserted:
            print("[patch-B] Could not find insertion point — Fix B skipped", flush=True)

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
