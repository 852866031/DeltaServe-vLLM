# eval/llama3

End-to-end evaluation harness for DeltaServe with **Meta-Llama-3-8B**. Supports two modes:

- **Inference only** — standard LoRA serving benchmark
- **Co-serving** (`--co`) — simultaneous inference + online SFT, where a LoRA adapter is updated in the background while requests are served

The harness was originally built for the RTX 5090; A100 timeline schedules are also checked in. GPU model is auto-detected at script load (`nvidia-smi`) and used to pick the right `timelines/<gpu>/` directory.

---

## Directory layout

```
eval/llama3/
├── init_adapters.py             # Train + initialize LoRA adapters (run once)
├── launch_llama3.py             # Thin wrapper that execs dserve.server.api_server
├── auto_benchmark.py            # Full benchmark orchestrator (launch → warmup → run → record)
├── auto_plot.py                 # Single-trace 4-panel plots for one config across modes
├── simple_test.py               # Single-request smoke test
├── analyze_finetuning_data.py   # Tokenize a dataset, print p50/p95/p99, recommend attn_bn_max / attn_l_max
├── keep_p95.py                  # Drop the top 5% longest samples so attn_l_max can be tightened
│
├── config/
│   ├── serving_config_finetuning.yaml      # Co-serving config (alpaca-1000 + packed_kv defaults)
│   └── serving_config_no_finetuning.yaml   # Inference-only config
│
├── adapters/
│   ├── llama3-toy-lora/         # Inference adapter (served at request time)
│   └── llama3-toy-lora-ft/      # Finetuning target adapter (updated during co-serving)
│
├── data/
│   ├── alpaca_1000_p95.txt      # Default FT dataset (alpaca-1000, top 5% length trimmed)
│   ├── alpaca_1000.txt          # Untrimmed alpaca-1000
│   ├── emotion.txt              # Legacy FT dataset (emotion classification prompts)
│   ├── load_alpaca.py           # Download + filter + flatten alpaca → alpaca_1000.txt
│   └── load_emotion.py          # Download HuggingFace emotion dataset → emotion.txt
│
├── timelines/
│   ├── 5090/                    # Request schedules tuned for RTX 5090
│   │   ├── timeline_live.csv    # Default schedule
│   │   ├── timeline_loose.csv   # Light load
│   │   ├── timeline_tight.csv   # Heavy load
│   │   └── timeline_nutanix.csv # Production trace (gitignored)
│   └── A100/                    # Same four files, tuned for A100
│
├── scripts/                     # Comparison/ablation plotting helpers
│   ├── bwd_graph_plot.py        # Eager vs. backward-graph comparison
│   ├── compare_graphs_plot.py   # Decode/prefill/bwd graph ablations
│   ├── compare_kv_plot.py       # unified vs. packed_kv allocator comparison
│   ├── compare_occupancy_plot.py # Page-occupancy curves (unified vs. packed_kv)
│   ├── compare_emotion_alpaca_plot.py # emotion vs. alpaca FT comparison
│   └── compare_allocators.py    # Sequential correctness check across allocators
│
├── output/                      # Run artifacts (per-batch CSVs, occupancy logs, scheduler stats)
└── plots/                       # PNG output for the plotting scripts above
```

`*.csv` is gitignored across the repo *except* for the timeline schedule files under `timelines/<gpu>/`, and the nutanix variant is gitignored unconditionally (proprietary trace).

---

## Quickstart

### 1. Install dependencies

```bash
pip install -U transformers datasets accelerate peft aiohttp pyyaml tqdm
huggingface-cli login   # required for Meta-Llama-3 weights
```

### 2. Initialize adapters

Trains a small LoRA adapter (Q/K/V/O projections, r=16) on a toy dataset and writes it to both adapter directories.

```bash
python init_adapters.py
```

### 3. (Optional) Refresh the finetuning dataset

The default config points at `data/alpaca_1000_p95.txt` (already checked in). To regenerate:

```bash
python data/load_alpaca.py        # downloads + filters → data/alpaca_1000.txt
python keep_p95.py --input data/alpaca_1000.txt --output data/alpaca_1000_p95.txt
```

### 4. Run a smoke test

Launches the server, waits for readiness, sends one request, then shuts down.

```bash
python simple_test.py [--co] [--port 9000]
```

### 5. Run the full benchmark

Launches the server, plays warmup requests, replays a timeline schedule, and writes per-request results.

```bash
# Inference only
python auto_benchmark.py

# Co-serving with all three CUDA graphs (decode + prefill + bwd)
python auto_benchmark.py --co --graphs

# Specific workload shape (tight / loose / nutanix)
python auto_benchmark.py --co --graphs --tight
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--timeline_csv` | `timelines/<gpu>/timeline_live.csv` | Request schedule to replay |
| `--timeline-gpu` | auto (`5090` or `A100`) | Which `timelines/<gpu>/` subdir to use; overridable on cross-GPU runs |
| `--loose` / `--tight` / `--nutanix` | off | Pick a named shape under the resolved timeline dir |
| `--base_model` | `meta-llama/Meta-Llama-3-8B` | HuggingFace model ID |
| `--lora_dir` | `adapters/llama3-toy-lora` | Adapter served at inference time |
| `--port` | `9000` | Server port |
| `--co` | off | Enable co-serving (FT runs in background) |
| `--decode_graph` | off | Enable decode CUDA graph capture |
| `--prefill_graph` | off | Enable prefill CUDA graph capture |
| `--bwd_graph` | off | Enable backward CUDA graph capture |
| `--graphs` | off | Shorthand for `--decode_graph --prefill_graph --bwd_graph` |
| `--track_occupancy` | off | Sample (used pages / total) at 1 Hz → `output/occupancy<suffix>.csv` |
| `--fold N` | 1 | Send every N-th request only (subsample timeline) |
| `--warmup_count` | 1000 | Cap on warmup rows (also capped by `--warmup_duration_s`) |
| `--warmup_duration_s` | 15 | Time window cap on warmup rows |
| `--warmup_rest_s` | 2 | Idle pause between warmup and the scheduled phase |
| `--out_csv` | `output/timeline_results<suffix>.csv` | Per-request results CSV |
| `--ft_log_path` | `output/bwd_log<suffix>.csv` | Backward-pass log CSV |

The `<suffix>` is auto-composed from which graph paths are enabled and which workload shape is active, e.g. `--co --graphs --tight` → `_decode_prefill_bwd_tight`. This keeps multiple runs from overwriting each other.

### 6. Plot results

Single-config 4-panel plot (one PNG per workload shape):

```bash
# After running --loose, --tight, and --nutanix
python auto_plot.py
```

Reads `output/timeline_results<suffix>_<mode>.csv` and `output/bwd_log<suffix>_<mode>.csv` plus the timeline schedule under `timelines/<gpu>/`. Subplots, left → right:

1. Scheduled request timeline
2. Per-request E2E latency vs. time
3. Throughput tokens/s (inference shaded under total; FT contribution is the gap)
4. Rolling TTFT SLO satisfaction rate (with 95% reference line; SLO read from the active YAML)

Comparison plots live under `scripts/`:

| Script | Purpose |
|---|---|
| `bwd_graph_plot.py` | Eager backward vs. graphed backward (1×3 layout) |
| `compare_graphs_plot.py` | Decode/prefill/bwd graph ablations (4-panel) |
| `compare_kv_plot.py` | `unified` vs. `packed_kv` allocator at the same workload |
| `compare_occupancy_plot.py` | Page-occupancy curves from `--track_occupancy` runs |
| `compare_emotion_alpaca_plot.py` | emotion vs. alpaca finetuning corpus |
| `compare_allocators.py` | Sequential correctness diff between allocators |

---

## Server configuration

`launch_llama3.py` execs `python -m dserve.server.api_server` and forwards two pieces:
- `--config <yaml>`: one of two files in `config/`
- `--override section.field=value`: zero or more YAML-typed overrides

YAML selection is binary:
- `--enable-finetuning` → `config/serving_config_finetuning.yaml` (alpaca + packed_kv bundled in as defaults)
- otherwise → `config/serving_config_no_finetuning.yaml`

`auto_benchmark.py` wraps this with the four graph/occupancy flags listed above. To exercise something the wrapper doesn't expose, drop down to `launch_llama3.py` directly and pass YAML `--override` flags via the CLI.

### Config sections (cheat sheet)

| YAML path | What it controls |
|---|---|
| `model.base_model` | HF model ID |
| `lora.adapter_dirs` | Inference adapter set (resolved to absolute paths at launch) |
| `finetune.data_path`, `finetune.lora_path` | FT corpus + target adapter |
| `finetune.num_epochs` | Epoch cap for the FT loop |
| `cuda_graph.enable_{decode,prefill,bwd}_cuda_graph` | Per-region graph toggles |
| `cuda_graph.use_graphed_bwd_attention` | Padded vs. monolithic attention bwd |
| `cuda_graph.attn_bn_max`, `cuda_graph.attn_l_max` | Padded-attn shape budget (`keep_p95.py` + `analyze_finetuning_data.py` size these) |
| `cuda_graph.prefill_sweep_max_tokens` | Upper bound on offline prefill profiling sweep |
| `cuda_graph.max_graph_memory_gb` | Cap on total decode+prefill graph memory (null/-1 = no cap; over-cap → run eager) |
| `memory.allocator` | `auto` (default — packed_kv on GQA), `unified`, or `packed_kv` |
| `memory.unified_mem_manager_log_path` | If non-null, write 1 Hz allocator occupancy CSV |
| `scheduler.batch_prediction_stats_path` | Where the BatchExecutionTracker dumps per-batch decisions at FT exit (default `output/scheduler/batch_prediction_stats.csv`; null disables) |
| `slo.{ttft,avg_tbt,max_tbt}_slo` | Scheduler admission gates in `mixed_req_queue.py` |

---

## SLO defaults

| Config | TTFT SLO | Avg TBT SLO | Max TBT SLO |
|---|---|---|---|
| `serving_config_finetuning.yaml` (co-serving) | 0.40 s | 0.15 s | 0.35 s |
| `serving_config_no_finetuning.yaml` (inference only) | 0.35 s | 0.15 s | 0.35 s |

These can be overridden per-run via `--override slo.ttft_slo=...` on `launch_llama3.py`, or by editing the YAMLs directly.

---

## Related docs

- `../../CLAUDE.md` (project root) — architecture overview, backward-path internals, memory allocator details, scheduler/estimator design.
- `../../MEMORY_ANALYSIS.md` — where each GB of GPU memory goes under the `--co --graphs` workload.
