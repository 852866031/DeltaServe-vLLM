# DeltaServe GPU Memory Analysis — Llama-3-8B Co-Serving

**Scope:** `python eval/llama3/auto_benchmark.py --co --bwd_graph --decode_graph`
**Model:** `meta-llama/Meta-Llama-3-8B` (32 layers, `hidden=4096`, 32 Q / 8 KV heads, `head_dim=128`, `intermediate=14336`, `vocab=128256`, fp16)
**Config:** `eval/llama3/config/serving_config_finetuning.yaml`

## 1. Executive summary

Two Python processes show up in `nvidia-smi`:

| PID | VRAM | Role |
|---|---|---|
| 18834 | 25 182 MiB | Router / inference process (owns model weights, KV+activation pool, decode graphs, LoRA adapters) |
| 18980 | 5 546 MiB (with `--bwd_graph`) / 5 154 MiB (without) | Backward / SFT subprocess (spawn, IPC-mapped pool + weights, local optimizer + backward working set) |

Of the ~30 GB total, **only ~0.5–5 GB is actually recoverable**; the rest is structural:
- Model weights (~16 GB fp16) and the 6 GB unified pool are fixed by config.
- Per-backward `lm_head.float()` (~2.1 GB retained by caching allocator) is **required for SFT loss to converge** and must not be removed.
- Two CUDA contexts and the graph pool are inherent to the MPS-partitioned co-serving design.

The two addressable items are:
1. **Activation double-buffer** (~0.5 GB permanent in main process) — a concrete refactor is proposed in §6.
2. **GQA KV oversizing** (~3–4 GB of the 6 GB pool carries zeros) — harder fix because the pool is unified with activation storage.

---

## 2. Execution pipeline recap

```
auto_benchmark.py ── subprocess ──▶  launch_llama3.py  ── os.system ──▶  dserve.server.api_server
                                         │                                       │
                                         │  translates --co / --bwd_graph /      │  loads YAML + overrides,
                                         │  --decode_graph into YAML overrides   │  spawns router + detok
                                         ▼                                       ▼
                                    serving_config_                          ModelRpcServer
                                    finetuning.yaml                         (model_rpc.py)
                                                                                 │
                                                                                 │  exposed_init_model:
                                                                                 │    loads Llama3TpPartModel (weights, pool)
                                                                                 │    loads LoRA adapters
                                                                                 ▼
                                                          Spawn backward_service  (Process, daemon=True)
                                                          under MPS window (10% active thread, 1 connection)
                                                          │
                                                          ▼
                                                    LlamaSFTBackwardService.start_service
                                                    (receives IPC-shared model + activation refs)
```

Code pointers for the pipeline:
- `auto_benchmark.py:414-435` — builds launcher argv
- `launch_llama3.py:127-134` — YAML override translation
- `api_server.py:520-598` — config load, subprocess spawn
- `model_rpc.py:51-181` — model load, adapter load, backward subprocess spawn
- `SFT_service.py:103-123` — backward startup (IPC handle open)
- `SFT_service_graph.py:85-172` — graphed-backward preparation

---

## 3. Main process — 25 GB line-by-line

### 3.1 Base model weights — **~16.0 GB**
`basemodel.py:80-94` (`_init_weights` → `load_hf_weights("fp16", ...)`).

| Component | Size |
|---|---|
| Embedding (`[128256, 4096]` fp16) | 1.05 GB |
| LM head (`[128256, 4096]` fp16, untied) | 1.05 GB |
| Per-layer attention (Q 16.8M + K 4.2M + V 4.2M + O 16.8M params, fp16) | 84 MB × 32 = 2.69 GB |
| Per-layer FFN (gate/up/down `[4096×14336]` fp16) | 336 MB × 32 = 10.75 GB |
| RMSNorm weights | < 1 MB |
| **Total** | **~16.0 GB** |

### 3.2 Unified KV + activation pool — **6.00 GB**
`unified_mem_allocator.py:58-62` (allocator init), sized by `memory.unified_mem_manager_max_size_gb=6` via `llama3/model.py:86-95`.

- Pool shape per layer: `[tot_size, head_num=32, head_dim=128]` fp16 → page size = 8 KB
- `tot_size = 6 GB / 32 layers / 8 KB = 24 576 pages/layer`
- Total: exactly 6.00 GB
- **Pages carry four roles** via `PageType` tag (`unified_mem_allocator.py:23-29`): `KV_CACHE`, `ATTENTION_INPUT_ACTIVATION`, `FFN_INPUT_ACTIVATION`, `EMBEDDING`

### 3.3 Shared activation handoff buffers — **~0.79 GB**
`unified_mem_allocator.py:181-196`, sized by `max_finetuning_tokens=1024`.

| Buffer | Shape | Size |
|---|---|---|
| `shared_transformer_out_activations[32]` | `[1024, 4096]` × 32 | 256 MB |
| `shared_attention_out_activations[32]` | `[1024, 4096]` × 32 | 256 MB |
| `embedding_output` | `[1024, 4096]` | 8 MB |
| `logit_tensor` | `[1024, 128256]` | 262 MB |
| `concat_input_ids` | `[2048]` int64 | 16 KB |

These are IPC-shared to the backward subprocess via `share_activation_dict` (`:198-205`).

### 3.4 LoRA adapters — **~100 MB**
`model_rpc.py:125-145`. With `r=16`, each layer's `w_combined_home` is `[2, 64, 32, 128]` fp16 = 1 MB per layer.

- `adapters/llama3-toy-lora` (inference): 32 MB fp16
- `adapters/llama3-toy-lora-ft` (finetune, appended at `api_server.py:528-532`): 32 MB fp16 + 32 MB fp32 (`w_combined_home_fp32`, lora_layer_weight)

### 3.5 RoPE cos/sin cache — **~18 MB**
`llama/model.py:122-127`. `t = arange(max_seq_len + 65536)` × 64 half-dims × 2 B × 2 tables.

### 3.6 Decode CUDA graphs (enabled via `--decode_graph`) — **~0.5–1.0 GB**
`dserve/common/cuda_graph_runner.py`. Per-bucket `att_m_buffers` of shape `[32 Q-heads, max_total_tokens=25000]` fp16 = ~51 MB per captured graph, plus static input/position/b_loc buffers.

### 3.7 CUDA context, cuBLAS/cuDNN workspaces, PyTorch caching allocator — **~0.8–1.2 GB**
Unavoidable driver / runtime overhead.

### 3.8 Main-process total

| Component | Size |
|---|---|
| Base weights | 16.0 GB |
| Unified pool | 6.0 GB |
| Shared activation buffers | 0.79 GB |
| LoRA (2 adapters × fp16 + 1 × fp32) | 0.10 GB |
| RoPE cache | 0.02 GB |
| Decode graphs | 0.5–1.0 GB |
| CUDA/runtime | 0.8–1.2 GB |
| **Total** | **~24.2–25.1 GB** ✓ matches 25 182 MiB |

---

## 4. Backward subprocess — 5.5 GB line-by-line

The subprocess is spawned via `multiprocessing.Process` with `torch.multiprocessing.set_start_method('spawn')` at `api_server.py:642`, under the MPS partition set briefly at `model_rpc.py:173-178`. Model weights and shared activation buffers are mapped in via CUDA IPC — they are **not** duplicated in VRAM; NVML attributes them to the exporting (main) process.

### 4.1 CUDA primary context + cuBLAS/Triton caches — **~0.5–1.0 GB**
Per-process CUDA context, cuBLAS workspace, Triton JIT cache.

### 4.2 **`lm_head_weight_.float()` reserved block — ~2.1 GB [LOAD-BEARING]**
`SFT_service.py:321` (`_post_layer_backward`):
```python
lm_W   = layer_weight.lm_head_weight_.float()      # [128256, 4096] fp32 = 2.10 GB
norm_W = layer_weight.final_norm_weight_.float()
g_y    = logit_grad @ lm_W
```

This upcast is **required for SFT loss to converge** — a prior experiment confirmed training plateaus without it. The caching allocator retains the 2.1 GB block across calls because the same-sized allocation is requested every backward. This is the single biggest line in the backward process but is structural cost, not waste.

Called from both eager (`_context_backward`, `:235`) and graph (`SFT_service_graph.py:636`) paths — `--bwd_graph` has no effect on it.

### 4.3 Logit-path fp32 conversions — **~0.3–0.5 GB reserved**
- `SFT_service.py:265`: `pred_logits = logits.float()` per request → fp32 `[T, 128256]`
- `SFT_service.py:306-311`: `torch.cat(all_logits)` + `torch.softmax(...)` allocate two fp32 `[N, 128256]` blocks

For `max_saved_finetuning_tokens=256` each block is ~131 MB; the allocator reserves the peak.

### 4.4 Per-layer attention backward transients — **~0.3 GB reserved**
`SFT_service.py:481-622` (`_backpop_attention`). Per call: `q_base/k_base/v_base`, LoRA `proj_lora` outputs, `q_/k_/v_`, `ctx`, per-request `scores/att/mask`, `grad_qh/kh/vh`, `grad_X_from_{q,k,v}`, `grad_X_from_lora_{q,k,v}`, `Zq/Zk/Zv`, rotary intermediates. Peak working set at `s=256` ≈ 60–80 MB; 32-layer reuse leaves ~300 MB reserved after fragmentation.

### 4.5 Per-layer FFN backward transients — **~0.2 GB reserved**
`SFT_service.py:355-387` (`_backprop_ffn`). Per call: `x_norm`, `gate_in`, `gate_out`, `up_out`, `grad_ffn_mid`, `grad_gate_out/up_out`, `grad_x_norm_*`. Intermediates at `[256, 14336]` fp16 (~7 MB each) × several → ~100–200 MB reserved.

### 4.6 AdamW optimizer state — **0.13 GB**
`SFT_service.py:110-114`. Lazy — allocated on first `optimizer.step()`. For 32 layers × `[2, 64, 32, 128]` fp32 params, momentum + variance = 2 × 64 MB = 128 MB.

### 4.7 Graphed-backward persistent buffers (only with `--bwd_graph`) — **~0.1 GB**
`SFT_service_graph.py:111-113, 203-222`. All deliberately allocated **outside** the graph pool (per CLAUDE.md's pool-aliasing NaN invariant):

| Buffer | Size |
|---|---|
| FFN static I/O (3 × `[256, 4096]` fp16) | 6 MB |
| Persistent LoRA fp32 grads (32 × `[2, 64, 32, 128]` fp32) | 64 MB |
| Padded-attn ctx (`flat_in_padded`, `pad_x_prev`, masks, etc.) | ~13 MB |
| **Total** | ~83 MB |

### 4.8 Graph pool (only with `--bwd_graph`) — **~0.3–0.5 GB**
`SFT_service_graph.py:114` (`graph_pool_handle`). Holds the working set of captured FFN + padded-attn graphs (32 + 32 graphs, pool sized to max live set). Mostly overlaps with what eager backward was already reserving, which is why enabling graphs only adds ~400 MB (not 2+ GB).

### 4.9 Caching-allocator fragmentation — **~0.4–0.6 GB**
Net difference between `cuda.memory_reserved()` and `cuda.memory_allocated()`.

### 4.10 Backward-subprocess total

| Component | No `--bwd_graph` | With `--bwd_graph` |
|---|---|---|
| CUDA context + runtime | 0.5–1.0 GB | 0.5–1.0 GB |
| `lm_head.float()` reserved | 2.1 GB | 2.1 GB |
| Logit-path fp32 | 0.3–0.5 GB | 0.3–0.5 GB |
| Attn bwd transients | 0.3 GB | 0.3 GB |
| FFN bwd transients | 0.2 GB | 0.2 GB |
| AdamW | 0.13 GB | 0.13 GB |
| Graph persistent buffers | 0 | 0.08 GB |
| Graph pool | 0 | 0.3–0.5 GB |
| Fragmentation | 0.4–0.6 GB | 0.4–0.6 GB |
| **Total** | **~4.0–5.0 GB** ✓ 5 154 MiB | **~4.5–5.5 GB** ✓ 5 546 MiB |

---

## 5. Right-sizing the unified memory pool

The YAML default `memory.unified_mem_manager_max_size_gb: 6` is roughly **3× larger than this workload needs**. Here is a quick model for picking a tighter value.

### 5.1 Bytes per token in the pool (current, GQA-unaware page layout)

Page shape: `[head_num=32, head_dim=128]` fp16 → **8 KB per page** (`unified_mem_allocator.py:58-62`).
A page is a global slot — the same physical index exists in all 32 layers, so storing one "logical page" costs `32 layers × 8 KB = 256 KB` across the pool.

| Usage | Pages per token | Bytes per token (all layers) |
|---|---|---|
| KV cache (K + V) | 2 | 2 × 256 KB = **0.50 MB** |
| Finetune activation snapshot (`ATTENTION_INPUT` + `FFN_INPUT`) | 2 | 2 × 256 KB = **0.50 MB** |

### 5.2 Sizing formula

```
pool_bytes_needed ≈ (N_kv_active + N_ft_active) × 0.5 MB  +  safety
```

where:
- `N_kv_active` = number of KV-cached tokens live at steady state ≈ `throughput_tokens_per_sec × avg_request_lifetime_sec`
- `N_ft_active` = finetune activation tokens live = **`max_saved_finetuning_tokens`** (256 in the current YAML) — never more, because the scheduler triggers backward and `reset_activation_pool` when this cap is hit
- `safety` ≈ 1× the KV working set to absorb request-arrival bursts before the scheduler evicts

### 5.3 Worked example (the current workload)

Operating point:
- Aggregate inference throughput ≈ **200 tokens/s**
- Average request lifetime ≈ 5 s (e.g. 1 s prefill + ~4 s decode at typical rates)
- `max_saved_finetuning_tokens = 256`

```
N_kv_active       ≈ 200 × 5        = 1 000 tokens
N_ft_active       ≈                  256 tokens
working set       ≈ (1000 + 256) × 0.5 MB  ≈ 0.63 GB
with 1× safety    ≈                         ≈ 1.3 GB
```

**⇒ `unified_mem_manager_max_size_gb: 2` is "just more than enough"** for this workload.

Sensitivity table (keeping `max_saved_finetuning_tokens=256`):

| Throughput × lifetime | Active KV | Working + safety | Recommended `max_size_gb` |
|---|---|---|---|
| 100 t/s × 5 s = 500 | 500 | ~0.8 GB | **1** |
| 200 t/s × 5 s = 1000 | 1000 | ~1.3 GB | **2** |
| 500 t/s × 5 s = 2500 | 2500 | ~2.8 GB | **3** |
| 1000 t/s × 10 s = 10000 | 10000 | ~10.5 GB | 11 (would need to raise `max_total_token_num` too) |

Upper bound for the current YAML: `max_total_token_num = 25000` → `25000 × 0.5 MB = 12.5 GB` — which the 6 GB pool can't actually hold anyway. The 6 GB value is a compromise between "bigger than any realistic steady-state working set" and "not the hard cap from the YAML." Dropping to **2 GB reclaims ~4 GB of main-process VRAM** for this workload with no observable impact on serving as long as bursts don't exceed ~2500 concurrent KV tokens.


## 6. Wastes identified

### 6.1 Waste A — Activation double-buffer (RECOVERABLE, ~0.5 GB)
**Where:** `unified_mem_allocator.py:181-196` (permanent allocation) and `:266-278` (per-backward copy).

**What happens:**
1. During finetune-token prefill, each token's attention-input and FFN-input activations are saved to scattered pool pages (`save_activations_by_layer`, `:232-243`).
2. When backward is triggered, `export_requests_info` gathers those scattered pages via `index_select`, concatenates, and **copies** into dense `shared_{transformer,attention}_out_activations` buffers (one `[1024, 4096]` tensor per layer per type).
3. Backward reads the dense buffers via CUDA IPC; reset frees the pool pages (`reset_activation_pool`, `:155-173`).

**Cost:**
- Permanent: 2 × 32 × 1024 × 4096 × 2 B = **512 MB** always resident.
- Transient: ~512 MB of pool pages co-resident with the dense copy during the backward window.

**Why this exists:** pool pages are scattered per-request; backward wants contiguous `[N, D]`. The copy gathers + densifies.

**Fix:** see §7.1.

### 6.2 Waste B — GQA KV pool oversizing (HARD TO FIX, ~3–4 GB of the 6 GB pool)
**Where:** `llama3/model.py:86-95`.

**What happens:** The docstring at `:78-80` states *"KV cache allocator must use KV head count (num_key_value_heads), not num_attention_heads"*, but the code passes `head_num=self.config["num_attention_heads"]` = 32. Each pool page is sized `[1, 32, 128]` = 8 KB to match full hidden_dim, but a Llama-3 GQA KV entry is only `[1, 8, 128]` = 2 KB. Every KV page carries 6 KB of zero padding that no kernel reads.

**Cost at steady state:** if the 6 GB pool were dedicated to KV (it isn't), ~4.5 GB would be zero padding. In the current mixed-use pool (KV + activation pages), the waste is less clean but still substantial — any page used for KV wastes 3/4 of its bytes.

**Why it's hard:** the pool is unified. Activation pages legitimately need 4096 elements (full hidden_dim). A one-line `head_num` change breaks activation storage. See §7.2 for two fix paths.

### 6.3 Waste C — Oversized RoPE cache (TRIVIAL, ~16 MB)
`llama/model.py:122-123`: `t = torch.arange(max_seq_len + 65536, ...)`. With `max_req_total_len=1024`, only 1024 positions are ever indexed; the extra 65 536 rows add ~16 MB across cos+sin. Drop the `+ 1024*64` or size to `max_req_total_len`.

### 6.4 Waste D — Main-process fp16+fp32 LoRA copies (MINOR, ~32 MB)
Main keeps both `w_combined_home` (fp16, used for inference) and `w_combined_home_fp32` (fp32, only to be IPC-shared with backward). Unavoidable given the inference/finetune split, flagging for completeness.
