# Building & Running vLLM on an RTX 5090 (Blackwell, sm_120)

A reproducible, from-scratch setup for the DeltaServe-vLLM fork on a single
**RTX 5090**. This is the *clean* path — it skips the dead ends we hit while
debugging. Total time is dominated by one CUDA download and a one-time
FlashInfer JIT compile on first run.

## TL;DR of why this is fiddly

The 5090 is consumer Blackwell = **compute capability sm_120**. Three things bite:

1. **FlashInfer needs CUDA ≥ 12.9 for sm_120.** It detects the CUDA version from
   whichever `nvcc` is first on `PATH`. If a system CUDA 12.8 shadows the conda
   one, FlashInfer silently fails to register the arch and then aborts with a
   misleading *"requires sm75 or higher"*.
2. **The conda toolchain's linker can't find `libcuda.so`** for FlashInfer's
   runtime JIT unless you point it at the CUDA stubs dir (`LIBRARY_PATH`).
3. **FlashInfer has no prebuilt cubins for sm_120**, so it JIT-compiles its
   kernels on first use (one-time cost). vLLM's *own* precompiled kernels do
   cover sm_120, so **no full vLLM source build is needed** — the precompiled
   (Python-only editable) install works.

Everything below makes the **conda env own its CUDA 13.0** for the session, so
none of this leaks into other projects or other machines.

## Prerequisites

- An RTX 5090 with a recent NVIDIA driver (CUDA 13–capable; we used driver
  `580.x`). Check: `nvidia-smi`.
- `conda`/`miniconda` installed.
- `git`. (`uv` is optional — commands below use `uv pip`, but plain `pip` works
  if you drop the `uv` prefix.)

---

## Step 1 — Create the conda environment

```bash
conda create -n dserve-vllm python=3.12 -y
conda activate dserve-vllm
```

## Step 2 — Install CUDA 13.0 toolkit *inside the env* (provides nvcc 13.0)

PyTorch wheels do **not** include `nvcc`; FlashInfer's JIT needs it, and it must
be **≥ 12.9** for sm_120. Install a matching CUDA 13.0 toolkit into the env:

```bash
conda install -c nvidia cuda-toolkit=13.0 -y
# lighter alternative if the full toolkit is too heavy:
# conda install -c nvidia cuda-nvcc=13.0 cuda-cudart-dev=13.0 -y

ls "$CONDA_PREFIX/bin/nvcc"   # confirm nvcc now exists inside the env
```

## Step 3 — Point the toolchain at the env's CUDA (run in every new shell)

These exports make the env's nvcc 13.0 win over any system CUDA and give the JIT
linker the `libcuda.so` stub. Run them **whenever you open a new shell** for this
env (before building or running):

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
hash -r                                  # clear bash's cached path to any old nvcc

which nvcc && nvcc --version             # MUST show $CONDA_PREFIX/bin/nvcc, release 13.0
```

Do not continue until `nvcc --version` reports **13.0 from the env path** — that
single check is what prevents the whole "sm75" cascade.

> **To avoid retyping each shell**, persist them to the env instead:
> ```bash
> conda env config vars set CUDA_HOME=$CONDA_PREFIX LIBRARY_PATH=$CONDA_PREFIX/lib/stubs
> conda activate dserve-vllm   # reactivate to apply
> ```
> (PATH precedence still needs the `export PATH=...; hash -r` line if a system
> CUDA is prepended in your `~/.bashrc`.)

## Step 4 — Clone and build vLLM (precompiled, editable)

Installs vLLM's precompiled kernels (which already cover sm_120), pulls
**torch cu130 (CUDA 13.0)** via `--torch-backend=auto`, and brings the pinned
`flashinfer-python` as a dependency. **No full source compile.**

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# for the real fork, check out your chosen release tag instead of main:
#   git checkout vX.Y.Z

VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

Post-install sanity (no GPU work):

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
# expect e.g.: torch 2.11.0+cu130 | cuda 13.0
python -c "import torch; print('cap', torch.cuda.get_device_capability())"
# expect: cap (12, 0)
```

## Step 5 — Smoke test (tiny model, triggers the one-time FlashInfer JIT)

```bash
rm -rf ~/.cache/flashinfer        # start clean
python -c "from vllm import LLM; print(LLM('facebook/opt-125m').generate('Hello'))"
```

The first run pauses while FlashInfer compiles sm_120 (`120f`) kernels into
`~/.cache/flashinfer`; later runs are fast. Generated text = the toolchain works.

## Step 6 — Experiment: run Llama-3-8B on one prompt

The weights are already in your HF cache (`HF_HOME=/mnt/storage/huggingface`,
model `meta-llama/Meta-Llama-3-8B`, all 4 shards present), so this loads locally
with no download.

```bash
export HF_HOME=/mnt/storage/huggingface   # already set in ~/.bashrc; explicit here for clarity
export HF_HUB_OFFLINE=1                    # use the local cache only, no network

python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B", gpu_memory_utilization=0.85)
params = SamplingParams(temperature=0.0, max_tokens=64)
out = llm.generate(["The capital of France is"], params)
print("=== generation ===")
print(out[0].outputs[0].text)
PY
```

Notes:
- `Meta-Llama-3-8B` is the **base** model (not Instruct), so use completion-style
  prompts like above rather than chat turns.
- bf16 weights are ~16 GB; the 5090's 32 GB is plenty. If you co-run other GPU
  work, lower `gpu_memory_utilization`.
- An Instruct variant (`meta-llama/Meta-Llama-3-8B-Instruct`) and several LoRA
  adapters also live under the same `HF_HOME` — handy later for exercising
  vLLM's multi-LoRA path during DeltaServe integration.

---

## Troubleshooting map

| Symptom | Cause | Fix |
|---|---|---|
| `FlashInfer requires GPUs with sm75 or higher` | `which nvcc` resolves to CUDA < 12.9 (system 12.8 shadowing the env) | Step 3 — make the env's nvcc 13.0 win; re-verify `nvcc --version` |
| `SM 12.x requires CUDA >= 12.9` | same as above (underlying error, usually swallowed) | same |
| `cannot find -lcuda` at JIT link | linker has no `libcuda.so` on its path | Step 3 — `LIBRARY_PATH=$CONDA_PREFIX/lib/stubs` |
| `sm_120 is not compatible with the current PyTorch` | torch isn't cu128+/cu130 | check `torch.version.cuda` ≥ 12.8 (Step 4); reinstall from a cu130 index if needed |
| `CUDA error: no kernel image is available` from a vLLM `_C` op | vLLM's *own* kernels lack sm_120 (shouldn't happen with current precompiled wheels) | only then: full source build with `TORCH_CUDA_ARCH_LIST=12.0`, no `VLLM_USE_PRECOMPILED` |
| model download / 401 on Step 6 | `HF_HUB_OFFLINE` unset and weights not found, or gated model | confirm `HF_HOME=/mnt/storage/huggingface` and set `HF_HUB_OFFLINE=1` |

---

## Portability note: A100 (and other GPUs)

- **The CUDA-version gymnastics here are 5090/Blackwell-specific.** The A100
  (sm_80) is the mainstream path: standard torch + prebuilt FlashInfer cubins,
  no JIT, no `-lcuda` step, no CUDA-≥12.9 requirement. Its setup is *simpler*,
  and vLLM auto-enables more optimizations there with no code change.
- **Keep arch-specific *runtime* overrides out of the committed fork and shared
  config.** In particular, do **not** hardcode `VLLM_ATTENTION_BACKEND` (or
  `enforce_eager`, FA-version pins, etc.) anywhere that ships to all machines —
  that would disable the faster auto-selected backend on the A100. vLLM picks
  the best backend per detected GPU; let it.
- The per-session / per-env CUDA approach (Step 3) means each machine sets up its
  own toolchain locally; nothing in this file needs to change in DeltaServe code
  to deploy on a different GPU.
