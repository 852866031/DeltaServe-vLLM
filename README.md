# DeltaServe-vLLM

`dserve-vllm` re-hosts DeltaServe's LLM **co-serving** layer on top of vLLM's
V1 engine: it interleaves a LoRA-SFT backward pass with ongoing inference on
the same GPU (activation capture → backward subprocess → SLO-aware
FT-admission scheduler), while keeping vLLM's production multi-LoRA batching
and multi-process engine as the substrate.

The Python distribution name and CLI are both `dserve-vllm`; internally it
still imports as `vllm` (it is a vendored fork of `vllm-project/vllm`), so
existing vLLM code and notebooks keep working.

> **On an RTX 5090?** Skip the general guide and jump straight to
> [**RTX 5090 (Blackwell, sm_120)**](#rtx-5090-blackwell-sm_120). The 5090
> needs CUDA ≥ 12.9 and a JIT step that the general path does not, so the
> 5090 section below is the *complete* installation guide for that GPU.

---

## What's in this repo

```
DeltaServe-vLLM/
├── dserve-vllm/                ← the package source (vendored vLLM fork)
│   ├── pyproject.toml          ← name = "dserve-vllm", CLI = "dserve-vllm"
│   └── vllm/deltaserve/        ← DeltaServe co-serving code (new)
├── DeltaServe/                 ← read-only reference (original framework)
├── configs/                    ← serving_config_finetuning_{llama3,opt}.yaml
├── scripts/                    ← ft_experiment_{llama3,opt}.py, launch_deltaserve.py
├── eval/                       ← auto_benchmark.py + timelines + plotting
├── adapters/                   ← toy LoRA adapters
└── tests/
```

Read after install:
- `CLAUDE.md` — architecture, box mapping, build/precision rules, current status.
- `INTEGRATION_PROGRESS.md` — phased plan + per-stage progress.
- `VLLM_FORK_CHANGES.md` — every change vs upstream vLLM (navigate the fork).

---

## Installation — general (most GPUs)

> **For RTX 5090 (Blackwell, sm_120)**: do **not** follow this section — use
> [**RTX 5090 (Blackwell, sm_120)**](#rtx-5090-blackwell-sm_120) instead, which
> is the same flow with the extra CUDA-13.0-in-env steps that Blackwell needs.

Works on Ampere (sm_80, e.g. A100), Ada (sm_89, e.g. L40/4090), and Hopper
(sm_90, e.g. H100). No nvcc-on-PATH gymnastics: PyTorch's bundled CUDA
runtime is enough and FlashInfer ships prebuilt cubins for these archs.

### Prerequisites
- A recent NVIDIA driver (CUDA 12.x-capable). Check: `nvidia-smi`.
- `conda` (or `miniconda`/`mambaforge`) and `git`.

### Steps

```bash
# 1. Clone
git clone https://github.com/<your-org>/DeltaServe-vLLM.git
cd DeltaServe-vLLM

# 2. Fresh conda env
conda create -n dserve-vllm python=3.12 -y
conda activate dserve-vllm

# 3. Install uv (lets us use --torch-backend=auto to pick the matching
#    torch wheel for your detected CUDA — saves you from hand-picking an
#    index URL). Skip if you already have it.
pip install uv

# 4. Editable install of the dserve-vllm package (precompiled wheel path —
#    fast, no full CUDA source build needed). Three env vars pin the install:
#    - VLLM_PRECOMPILED_WHEEL_COMMIT pins the `.so` files to the upstream
#      vLLM commit this fork was vendored from (without it, git merge-base
#      can't find a base commit because our main shares no history with
#      vllm-project/vllm:main, and the install falls back to a nightly wheel
#      that may have ABI-incompatible vllm._C symbols).
#    - VLLM_VERSION_OVERRIDE pins the package version string. setuptools-scm
#      can't derive one (no tags in this repo, no shared history with
#      upstream's tags); without the override the install errors. The value
#      below matches what `pip show vllm` reported on this fork before the
#      package rename.
#    - VLLM_USE_PRECOMPILED=1 enables the wheel-grafting code path.
cd dserve-vllm
VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
    VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 \
    VLLM_USE_PRECOMPILED=1 \
    uv pip install --editable . --torch-backend=auto
cd ..
```

> **Tip — persist the env vars** so you don't have to retype them on the next
> install (and so any subprocess you spawn in this env inherits them):
> ```bash
> conda env config vars set \
>     VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
>     VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 \
>     VLLM_USE_PRECOMPILED=1
> conda activate dserve-vllm   # reactivate to apply
> ```
> After this, the install command collapses to:
> `uv pip install --editable . --torch-backend=auto`.

> **Without `uv`** — plain `pip` does not support `--torch-backend`, so install
> torch from the right index first, then do the editable install without the
> flag:
> ```bash
> # pick the index that matches your CUDA major version:
> #   CUDA 12.x → cu128    CUDA 13.x → cu130
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> cd dserve-vllm
> VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
>     VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 \
>     VLLM_USE_PRECOMPILED=1 \
>     pip install --editable .
> cd ..
> ```

### Sanity check (no GPU work)

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
python -c "import torch; print('device cap', torch.cuda.get_device_capability())"
dserve-vllm --help          # CLI registered? Should print vLLM subcommands.
```

### Smoke test (tiny model on GPU)

```bash
python -c "from vllm import LLM; print(LLM('facebook/opt-125m').generate('Hello'))"
```

Generated text → the install works. Skip ahead to
[**Running the experiments**](#running-the-experiments).

---

## RTX 5090 (Blackwell, sm_120)

A reproducible, from-scratch setup on a single RTX 5090. This is the *clean*
path — it skips the dead ends we hit while debugging. Total time is dominated
by one CUDA download and a one-time FlashInfer JIT compile on first run.

### Why this is fiddly

The 5090 is consumer Blackwell = **compute capability sm_120**. Three things
bite:

1. **FlashInfer needs CUDA ≥ 12.9 for sm_120.** It detects the CUDA version
   from whichever `nvcc` is first on `PATH`. If a system CUDA 12.8 shadows
   the conda one, FlashInfer silently fails to register the arch and then
   aborts with a misleading *"requires sm75 or higher"*.
2. **The conda toolchain's linker can't find `libcuda.so`** for FlashInfer's
   runtime JIT unless you point it at the CUDA stubs dir (`LIBRARY_PATH`).
3. **FlashInfer has no prebuilt cubins for sm_120**, so it JIT-compiles its
   kernels on first use (one-time cost). vLLM's *own* precompiled kernels do
   cover sm_120, so **no full source build is needed** — the precompiled
   (Python-only editable) install works.

Everything below makes the **conda env own its CUDA 13.0** for the session,
so none of this leaks into other projects or other machines.

### Prerequisites

- An RTX 5090 with a recent NVIDIA driver (CUDA 13–capable; we used driver
  `580.x`). Check: `nvidia-smi`.
- `conda`/`miniconda` installed.
- `git`. (`uv` is optional — commands below use `pip`, but `uv pip` is faster
  if installed.)

### Step 1 — Clone the repo

```bash
git clone https://github.com/<your-org>/DeltaServe-vLLM.git
cd DeltaServe-vLLM
```

### Step 2 — Create the conda environment

```bash
conda create -n dserve-vllm python=3.12 -y
conda activate dserve-vllm
```

### Step 3 — Install CUDA 13.0 toolkit *inside the env* (provides nvcc 13.0)

PyTorch wheels do **not** include `nvcc`; FlashInfer's JIT needs it, and it
must be **≥ 12.9** for sm_120. Install a matching CUDA 13.0 toolkit into the
env:

```bash
conda install -c nvidia cuda-toolkit=13.0 -y
# lighter alternative if the full toolkit is too heavy:
# conda install -c nvidia cuda-nvcc=13.0 cuda-cudart-dev=13.0 -y

ls "$CONDA_PREFIX/bin/nvcc"   # confirm nvcc now exists inside the env
```

### Step 4 — Point the toolchain at the env's CUDA (run in every new shell)

These exports make the env's nvcc 13.0 win over any system CUDA and give the
JIT linker the `libcuda.so` stub. Run them **whenever you open a new shell**
for this env (before building or running):

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
hash -r                                  # clear bash's cached path to any old nvcc

which nvcc && nvcc --version             # MUST show $CONDA_PREFIX/bin/nvcc, release 13.0
```

Do not continue until `nvcc --version` reports **13.0 from the env path** —
that single check is what prevents the whole "sm75" cascade.

> **To avoid retyping each shell**, persist them to the env instead:
> ```bash
> conda env config vars set CUDA_HOME=$CONDA_PREFIX LIBRARY_PATH=$CONDA_PREFIX/lib/stubs
> conda activate dserve-vllm   # reactivate to apply
> ```
> (PATH precedence still needs the `export PATH=...; hash -r` line if a
> system CUDA is prepended in your `~/.bashrc`.)

### Step 5 — Install dserve-vllm (precompiled, editable)

Installs the package with vLLM's precompiled kernels (which already cover
sm_120), pulls **torch cu130 (CUDA 13.0)** via `--torch-backend=auto`, and
brings the pinned `flashinfer-python` as a dependency. **No full source
compile.**

Three env vars pin the install (same as the
[general install](#installation--general-most-gpus); see that section for the
full rationale):

- **`VLLM_VERSION_OVERRIDE`** — pins the package version string. setuptools-scm
  can't derive one for this repo (no tags, no shared history with upstream's
  tags), so without the override the install errors. The value below matches
  what `pip show vllm` reported on this fork before the package rename.
- **`VLLM_PRECOMPILED_WHEEL_COMMIT`** — pins the `.so` files to the upstream
  vLLM commit this fork was vendored from
  ([`117afeea4`](https://github.com/vllm-project/vllm/commit/117afeea4)).
  Without it the install falls back to a nightly wheel with possibly
  ABI-incompatible `vllm._C` symbols.
- **`VLLM_USE_PRECOMPILED=1`** — enables the wheel-grafting code path (no
  full source build).

```bash
# Need uv for --torch-backend=auto (plain pip doesn't have this flag).
pip install uv

cd dserve-vllm
VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
    VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 \
    VLLM_USE_PRECOMPILED=1 \
    uv pip install --editable . --torch-backend=auto
cd ..
```

> **Tip — persist the env vars** so you don't have to retype them on the next
> install:
> ```bash
> conda env config vars set \
>     VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
>     VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 \
>     VLLM_USE_PRECOMPILED=1
> conda activate dserve-vllm   # reactivate to apply
> ```
> After this, the install command collapses to:
> `uv pip install --editable . --torch-backend=auto`.

> **Without `uv`** — install torch cu130 from PyTorch's index first, then do
> the editable install without `--torch-backend`:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu130
> cd dserve-vllm
> VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
>     VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 \
>     VLLM_USE_PRECOMPILED=1 \
>     pip install --editable .
> cd ..
> ```

> **If you bump the vendoring base** — get the new 40-char SHA from
> `git log --all --grep "vendored fork"` (or whichever commit you rebased on)
> and update the env var above. The wheel must exist at
> `https://wheels.vllm.ai/<sha>/cu130/vllm/metadata.json` — a quick
> `curl -I` will confirm.

Post-install sanity (no GPU work):

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
# expect e.g.: torch 2.11.0+cu130 | cuda 13.0
python -c "import torch; print('cap', torch.cuda.get_device_capability())"
# expect: cap (12, 0)
dserve-vllm --help          # CLI registered
```

### Step 6 — Smoke test (tiny model, triggers the one-time FlashInfer JIT)

```bash
rm -rf ~/.cache/flashinfer        # start clean
python -c "from vllm import LLM; print(LLM('facebook/opt-125m').generate('Hello'))"
```

The first run pauses while FlashInfer compiles sm_120 (`120f`) kernels into
`~/.cache/flashinfer`; later runs are fast. Generated text → the toolchain
works.

### Step 7 — Real-model check (Llama-3-8B on one prompt)

```bash
export HF_HOME=/path/to/your/huggingface/cache  # or wherever you store weights
export HF_HUB_OFFLINE=1                          # use the local cache only

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
- `Meta-Llama-3-8B` is the **base** model (not Instruct), so use
  completion-style prompts like above rather than chat turns.
- bf16 weights are ~16 GB; the 5090's 32 GB is plenty. If you co-run other
  GPU work, lower `gpu_memory_utilization`.

---

### Troubleshooting map (RTX 5090)

| Symptom | Cause | Fix |
|---|---|---|
| `FlashInfer requires GPUs with sm75 or higher` | `which nvcc` resolves to CUDA < 12.9 (system 12.8 shadowing the env) | Step 4 — make the env's nvcc 13.0 win; re-verify `nvcc --version` |
| `SM 12.x requires CUDA >= 12.9` | same as above (underlying error, usually swallowed) | same |
| `cannot find -lcuda` at JIT link | linker has no `libcuda.so` on its path | Step 4 — `LIBRARY_PATH=$CONDA_PREFIX/lib/stubs` |
| `sm_120 is not compatible with the current PyTorch` | torch isn't cu128+/cu130 | check `torch.version.cuda` ≥ 12.8 (Step 5); reinstall from a cu130 index if needed |
| `CUDA error: no kernel image is available` from a vLLM `_C` op | vLLM's *own* kernels lack sm_120 (shouldn't happen with current precompiled wheels) | only then: full source build with `TORCH_CUDA_ARCH_LIST=12.0`, no `VLLM_USE_PRECOMPILED` |
| model download / 401 on Step 7 | `HF_HUB_OFFLINE` unset and weights not found, or gated model | confirm `HF_HOME` points at your cache and set `HF_HUB_OFFLINE=1` |

### Portability note: A100 (and other GPUs)

The CUDA-version gymnastics above are **5090/Blackwell-specific**. The A100
(sm_80) and other Ampere/Ada/Hopper GPUs are the mainstream path: standard
torch + prebuilt FlashInfer cubins, no JIT, no `-lcuda` step, no CUDA-≥12.9
requirement. Their setup is the
[general install](#installation--general-most-gpus) above. vLLM auto-enables
more optimizations there with no code change.

Keep arch-specific *runtime* overrides out of the committed configs. In
particular, do **not** hardcode `VLLM_ATTENTION_BACKEND` (or `enforce_eager`,
FA-version pins, etc.) anywhere that ships to all machines — that would
disable the faster auto-selected backend on the A100. vLLM picks the best
backend per detected GPU; let it.

---

## Running the experiments

The headline eval is `eval/auto_benchmark.py` — a timeline-replay benchmark
that launches a `dserve-vllm serve` server with a Llama-3 base + two LoRA
adapters and measures TTFT / TBT / FT throughput under a recorded inference
load. To run it you first need the two adapter directories that the eval's
finetuning YAML references.

### 1. Build the toy Llama-3 LoRA adapters

`scripts/init_adapters_llama3.py` trains a tiny rank-16 Q/K/V/O LoRA on
Llama-3-8B against a handful of toy prompts (a few minutes on one 5090), then
saves it to **two** sibling directories that the eval expects:

| Dir | Role at eval time |
|---|---|
| `adapters/llama3-toy-lora` | The **inference** adapter — served via vLLM's normal multi-LoRA path. Every benchmark request targets this adapter by name. |
| `adapters/llama3-toy-lora-ft` | The **finetuning** target — same weights at init time. During co-serving, DeltaServe's backward subprocess keeps training this one further; inference is unaffected because the served adapter is the other dir. |

The script is committed (`scripts/init_adapters_llama3.py`), so you only need
to run it. From the repo root, inside the `dserve-vllm` conda env:

```bash
# One-time: log in so HuggingFace can fetch the gated Llama-3-8B weights.
huggingface-cli login

# Train + save. Single-GPU is fine; multi-GPU via accelerate is faster:
python scripts/init_adapters_llama3.py
# or:
accelerate launch --multi_gpu scripts/init_adapters_llama3.py

# Idempotent re-run guard (no-op if both dirs already exist):
python scripts/init_adapters_llama3.py --skip-if-exists
```

Useful flags (`--help` for the full list):
- `--out-dir DIR` — write to `DIR/llama3-toy-lora{,-ft}` instead of the
  default `<repo>/adapters/...`. The eval reads the default location, so
  override only if you're staging adapters somewhere else.
- `--epochs N` — defaults to 2 to keep wall time small; the adapter is
  intentionally tiny because the eval doesn't care about its quality.

When the script finishes, `ls adapters/llama3-toy-lora{,-ft}` should each show
`adapter_config.json`, `adapter_model.safetensors`, and a tokenizer.

### 2. Run the timeline benchmark

`eval/auto_benchmark.py` does the following end-to-end, in one invocation:

1. **Spawns the server** — `dserve-vllm serve meta-llama/Meta-Llama-3-8B`
   with the eval YAML (`configs/serving_config_finetuning_llama3.yaml`), the
   inference LoRA module wired in by name, and (under `--co`) the
   FinetuneConfig + SLO section that enable co-serving. Server logs stream
   live and are also captured to `eval/output/server<suffix>.log`.
2. **Waits for `/health`**, runs a **warmup** phase (first ~1k timeline rows,
   replayed but **not recorded**) so cold-start FlashInfer JIT compiles +
   KV-cache warmups don't pollute the measurements.
3. **Replays the timeline** — a recorded inference trace from
   `eval/timelines/<gpu>/timeline_<mode>.csv` (auto-selected per
   `nvidia-smi`; available timelines: `loose`, `tight`, `live`). Each row
   becomes a **streaming** `POST /v1/completions` call; the script records
   `ttft_s` (time-to-first-token), `latency_s` (last token), and per-request
   avg/worst TBT from inter-chunk gaps.
4. **Triggers FT** via `POST /start_finetuning` under `--co` (the YAML holds
   admission closed at launch so warmup is FT-free); afterward, trims the
   server's `bwd_log<suffix>.csv` to the timeline window so plots align.
5. **Writes results** to `eval/output/timeline_results<suffix>.csv` (per-req
   metrics) and, under `--co`, `eval/output/bwd_log<suffix>.csv` (FT
   throughput log).

#### Common invocations

```bash
# Co-serving on the "loose" arrival pattern (the headline figure)
python eval/auto_benchmark.py --co --loose

# Inference-only baseline on the same arrival pattern — use to compare TTFT
# delta caused by FT contention vs. pure serving.
python eval/auto_benchmark.py --loose

# "tight" pattern — denser arrivals, stress-tests the SLO-aware admission gate
python eval/auto_benchmark.py --co --tight

# Live (default) pattern, finetuning off
python eval/auto_benchmark.py
```

Mode flags (`--loose` / `--tight` / `--nutanix`) are mutually exclusive and
just pick which `timeline_<mode>.csv` to replay. Without any, the script
loads `timeline_live.csv`. `--co` is independent — toggle it to compare
co-serving vs. inference-only on the same arrival pattern.

Output suffix convention: `<co?>_<mode>` — e.g. `--co --loose` produces
`timeline_results_co_loose.csv` and `bwd_log_co_loose.csv`. The plotter
(`eval/auto_plot.py`) reads these files to render the 4-panel
latency / throughput / SLO figure.

#### Other useful flags
- `--warmup_count N` / `--warmup_duration_s S` — tune the warmup phase
  (default: 1000 reqs or 10 s, whichever ends first).
- `--api-server-count N` — shard the OpenAI frontend across N processes
  (shared single EngineCore). Useful when the frontend asyncio loop becomes
  the bottleneck (Phase-4 finding); default `null` = 1 frontend.
- `--timeline-gpu {5090,A100}` — override the GPU-subdir auto-detection if
  you want to replay an A100-recorded trace on a 5090 or vice versa.

### Note on a fresh clone

Compiled kernels (`*.so`) and model weights (`*.safetensors`, tokenizer
blobs) are **gitignored**. After cloning you need to:
1. Reinstall the package (`VLLM_USE_PRECOMPILED=1 uv pip install -e .` from
   `dserve-vllm/`, with the env vars from [Step 5](#step-5--install-dserve-vllm-precompiled-editable)).
2. Re-run `scripts/init_adapters_llama3.py` to rebuild the adapter dirs.
3. Then `python eval/auto_benchmark.py --co --loose`.
