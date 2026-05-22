# Refactor: Rename to DeltaServe

**Branch:** `refactor/rename-to-deltaserve`
**Base branch:** `main`

---

## Overview

This branch performs a structural refactor of the repository: renaming the Python package, reorganising directories, eliminating hardcoded paths, and improving the evaluation tooling. No algorithmic changes are made to the inference or finetuning logic.

---

## Changes

### 1. Package rename: `slora` → `dserve`

The Python package was renamed from `slora` to `dserve` to reflect the project's identity as DeltaServe.

- **Before:** `S-LoRA/slora/` (nested under the old S-LoRA subdirectory)
- **After:** `dserve/` (at the repo root)

All internal imports across the codebase were updated accordingly (`from slora.x import y` → `from dserve.x import y`).

---

### 2. Directory restructure

The top-level layout was flattened. The `S-LoRA/` subdirectory is gone.

| Old path | New path |
|---|---|
| `S-LoRA/slora/` | `dserve/` |
| `S-LoRA/test/llama3/` | `eval/llama3/` |
| `S-LoRA/test/kernel/` | `test/kernel/` |
| `S-LoRA/test/model/` | `test/model/` |
| `S-LoRA/setup.py` | `setup.py` |
| `S-LoRA/clean.sh` | `clean.sh` |
| `S-LoRA/docs/` | `dev_docs/` |

Evaluation/experiment harnesses (`eval/`) are now separated from unit and kernel tests (`test/`).

---

### 3. Hardcoded path elimination

All scripts previously contained absolute paths tied to a specific machine and had to be manually edited before use. Every script now resolves its own paths using:

```python
from pathlib import Path
_HERE = Path(__file__).resolve().parent
```

Affected files:
- `eval/llama3/launch_llama3.py` — adapter dirs, config paths, log paths
- `eval/llama3/auto_benchmark.py` — timeline CSV, results CSV output
- `eval/llama3/auto_plot.py` — all four input/output CSV and PNG paths
- `eval/llama3/config/finetuning_config.json` / `no_finetuning_config.json` — paths are rewritten at runtime by `launch_llama3.py` via `update_json_paths()`

---

### 4. Adapter initialisation: `init_adapters.py`

Replaced the old two-step workflow:
```bash
# old
python adapter_train.py
cp -r adapters/llama3-toy-lora adapters/llama3-toy-lora-ft
```

With a single script that handles the full lifecycle:
```bash
# new
python init_adapters.py
```

What `init_adapters.py` does that the old workflow did not:
- Deletes `adapters/llama3-toy-lora/` and `adapters/llama3-toy-lora-ft/` before training, ensuring a clean slate
- Removes all `checkpoint-*` subdirectories written by the Hugging Face `Trainer` after the final adapter is saved (prevents large intermediate files from accumulating or being committed)
- Copies the final adapter to `llama3-toy-lora-ft` automatically

---

### 5. Adapter training memory efficiency

The adapter training in `init_adapters.py` was updated to avoid CUDA out-of-memory errors on single-GPU machines when training LLaMA-3-8B:

- **`device_map="auto"`** — distributes model layers across all available GPUs (and CPU if needed) rather than loading everything onto GPU 0
- **`gradient_checkpointing=True`** — recomputes activations during the backward pass instead of holding them all in memory; trades a small amount of compute for a significant reduction in peak activation memory
- **`model.enable_input_require_grads()`** — required when combining PEFT/LoRA adapters with gradient checkpointing so gradients flow correctly through the frozen base layers

---

### 6. Documentation

- **`eval/llama3/README.md`** (new) — full reference for the LLaMA-3 evaluation harness: directory layout, quickstart, `auto_benchmark.py` argument table, server configuration, and SLO thresholds
- **`README.md`** (updated) — removed stale `S-LoRA/` paths, removed the manual path-editing instructions, updated all commands to use the new `eval/llama3/` layout
- **`eval/llama3/adapters/.gitignore`** (new) — ignores `checkpoint-*/` directories so Trainer checkpoints are never accidentally committed

---

## Files not changed

- All inference and finetuning logic under `dserve/` is unchanged from the original `S-LoRA/slora/` — only imports and file locations differ
- Kernel tests under `test/` are unchanged
- `setup.py`, `environment.yml`, `LICENSE` are unchanged
