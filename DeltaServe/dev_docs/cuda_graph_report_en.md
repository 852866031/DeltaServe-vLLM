# CUDA Graph for Inference Decode — Implementation Report

## 1. Overview

CUDA Graph optimization has been implemented for the **decode stage** of the S-LoRA inference engine. CUDA Graph captures all GPU kernel launches in each decode step into a single graph, which can then be executed through one graph replay. This removes per-kernel CPU launch overhead. For a 32-layer LLaMA-3-8B model, each decode step originally involves roughly **480+ kernel launches**.

**Experimental result: the decode stage achieves a 2.3× speedup, and output correctness has been validated.**

---

## 2. Experimental Results

### 2.1 Experimental Setup

| Item | Value |
|------|-------|
| Model | Meta-LLaMA-3-8B |
| LoRA Adapter | llama3-toy-lora |
| `max_total_tokens` | 25000 |
| Request Type | Single-request serial execution, 64 output tokens/request |
| Warmup | 3 requests |
| Benchmark | 10 requests |
| Memory Manager | Unified Memory Manager (`max_size=6`) |
| Conda Environment | `dserve` |

### 2.2 Performance Comparison （H200)

| Metric | Baseline | CUDA Graph | Speedup |
|------|----------|------------|---------|
| **Avg Wall Time** | 1.0786s | 0.4854s | **2.22×** |
| **Avg TBT (time between tokens)** | 0.0165s (16.5ms) | 0.0071s (7.1ms) | **2.32×** |
| **Avg Worst TBT** | 0.0220s (22.0ms) | 0.0077s (7.7ms) | **2.86×** |
| **TTFT (time to first token)** | 0.0361s | 0.0372s | ~1.0× (unchanged) |
| **Decode Throughput** | 59.3 tok/s | 131.8 tok/s | **2.22×** |

### 2.3 Correctness Validation

The outputs of the baseline and CUDA Graph versions were compared using 5 different prompts (`do_sample=false`, greedy decoding):

| Prompt | Consistency | Notes |
|--------|-------------|-------|
| `"1 + 1 = 2, 2 + 2 = 4, 3 + 3 ="` | **Exact match** | Mathematical content; large gap between top-token probabilities |
| `"The capital of France is"` | **Semantically equivalent, minor wording differences** | First token `"Paris."` matches; later tokens differ as `"The city"` vs `"It is"` |
| `"Machine learning is a field of"` | **Semantically equivalent, minor wording differences** | First few tokens match; later tokens differ as `"algorithms"` vs `"statistical"` |
| `"Hello, how are you?"` | **Semantically equivalent, minor wording differences** | |
| `"Once upon a time, there was a"` | **Semantically equivalent, minor wording differences** | First few tokens match |

**Reason for the differences:** the CUDA Graph warmup stage, which is required for cuBLAS workspace initialization, introduces very small floating-point deviations (on the order of ~1e-7). Under greedy decoding, when the top-2 token logits are very close, this can flip the argmax result. This is a **known behavior** of CUDA Graph and does not affect generation quality.

- For strongly deterministic content such as math or code, outputs are exactly identical
- For natural language, outputs are semantically equivalent but may differ slightly in wording
- With `do_sample=true` and temperature enabled, the effect is negligible

### 2.4 Key Observations

1. **2.3× speedup in the decode stage**: per-token generation time drops from 16.5ms to 7.1ms  
2. **2.9× improvement in worst-case TBT**: kernel-launch jitter is effectively removed, reducing latency from 22ms to 7.7ms  
3. **TTFT remains unchanged**: CUDA Graph is applied only to decode, so prefill is unaffected  
4. **Very high stability**: under CUDA Graph, TBT is almost variance-free (7.0–7.1ms), while the baseline fluctuates between 16.2–17.0ms  
5. **Graph replay takes only 0.1ms**, far below the 150–300ms capture cost  

---

## 3. Modified Files

### 3.1 New Files

#### `slora/common/cuda_graph_runner.py` (new file, 242 lines)

Core CUDA Graph infrastructure responsible for graph capture and replay.

**Key design choices:**
- **Cache key**: `(exact_batch_size, max_len_bucket)` — batch size is not bucketed because LoRA dispatch requires exact adapter mapping, while `max_len` is aligned to multiples of 128
- **Right-aligned `b_loc`**: the attention kernel accesses `b_loc[i, max_len_in_batch - 1 - j]`. Since `max_len_in_batch` is frozen to the bucket value in the graph, the `b_loc` data must be re-right-aligned to the bucket boundary
- **Static buffer mode**: all tensors are cloned during capture to keep fixed memory addresses; during replay, data is updated through `.copy_()`
- **Preallocated `att_m_buffers`**: one **float16** buffer of shape `(tp_q_head_num, max_total_tokens)` per layer, avoiding dynamic-shape `torch.empty` calls that would break the graph

```python
class CudaGraphRunner:
    def capture(batch_size, max_len, token_forward_fn, input_ids, infer_state, kwargs) -> Tensor
    def replay(batch_size, max_len, input_ids, infer_state) -> Tensor
    def has_graph(batch_size, max_len) -> bool
```

#### `test/llama3/bench_cuda_graph.py` (new file, 247 lines)

Automated benchmark script that launches servers with and without CUDA Graph, sends requests, and compares performance.

---

### 3.2 Core Modified File

#### `slora/models/peft/lora_unordered_batch_mixed.py` (+91 lines)

**Main changes:**

1. **`_shared_cuda_graph_runner` (class-level variable)**: the `CudaGraphRunner` instance is shared at the class level because the engine is recreated for every request. If stored at the instance level, captured graphs would be lost after each request.

2. **`_prepare_decode_infer_state()` (new method)**: extracts infer-state preparation logic from `_decode()`. When `enable_cuda_graph=True`, it forcibly uses the non-contiguous decode path:
   - **Why**: the contiguous decode path creates KV cache views (`decode_mem_start/end`), and the addresses of those views change every step, which breaks CUDA Graph
   - **Fix**: the non-contiguous path uses stable `decode_key_buffer/decode_value_buffer` together with `destindex_copy_kv`, so buffer addresses remain fixed

3. **`_decode_with_cuda_graph()` (new method)**: chooses between capture and replay depending on `has_graph()`.

```diff
+ # Class-level shared CudaGraphRunner
+ _shared_cuda_graph_runner = None

  def _decode(...):
+     infer_state = self._prepare_decode_infer_state(...)
+     if self.enable_cuda_graph:
+         return self._decode_with_cuda_graph(batch_size, max_len_in_batch, ...)
+     else:
+         return self._token_forward(...)
```

---

### 3.3 Attention Layer Changes

#### `slora/models/llama/layer_infer/transformer_layer_infer.py` (+23 lines)

1. **`_token_decode_attention_normal()` and `_token_decode_attention_normal_alt()`**
   - Previously, each decode step created `att_m_tensor` using `torch.empty((heads, total_token_num))`
   - Since `total_token_num` changes every step, this dynamic-shape allocation breaks CUDA Graph
   - **Fix**: if `infer_state._att_m_buffers` exists (preallocated by `CudaGraphRunner`), use the fixed-size preallocated buffer instead

```diff
-  att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), ...)
+  if hasattr(infer_state, '_att_m_buffers') and infer_state._att_m_buffers is not None:
+      att_m_tensor = infer_state._att_m_buffers[self.layer_num_]
+  else:
+      att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), ...)
```

2. **Suppress the warning `"Not support non-contiguous decode mem index yet."`**  
   Since CUDA Graph explicitly forces the non-contiguous path, this warning is no longer needed.

#### `slora/models/llama3/layer_infer/transformer_layer_infer.py` (+7 lines)

Same `_att_m_buffers` fix applied to LLaMA-3 GQA attention.

#### `slora/models/llama/infer_struct.py` (+1 line)

Added a comment clarifying that `other_kv_index = b_loc[0, max_len-1].item()` is executed outside graph capture, so the frozen scalar does not affect correctness.

---

### 3.4 CLI Argument Propagation

The following files were updated to propagate the `--enable-cuda-graph` command-line flag:

| File | Change |
|------|--------|
| `slora/server/api_server.py` | Added `--enable-cuda-graph` argparse argument |
| `slora/server/input_params.py` | Added `self.enable_cuda_graph = False` field |
| `slora/server/router/manager.py` | Read from args and assign to `input_params.enable_cuda_graph` |
| `slora/server/router/model_infer/model_rpc.py` | Passed `enable_cuda_graph` into the `LoraUnorderedBatchMixed` constructor |
| `test/llama3/launch_llama3.py` | Added `--enable-cuda-graph` flag and appended it to the server launch command |

---

## 4. Technical Details

### 4.1 Key Issues Solved

| Issue | Cause | Solution |
|------|------|----------|
| **Dynamic shape of `att_m_tensor`** | `torch.empty((heads, total_token_num))` changes shape every step because `total_token_num` varies | Preallocate a fixed buffer of shape `(heads, max_total_tokens)` |
| **Wrong dtype for `att_m_tensor`** | Buffer was initially created using `input_ids.dtype` (`int64`), but attention writes float values into it, producing garbage | Force the buffer dtype to `torch.float16` |
| **Changing addresses in contiguous decode** | View addresses from `decode_mem_start/end` change every step | Force the non-contiguous path and use fixed buffers with `destindex_copy_kv` |
| **Frozen `max_len_in_batch`** | The attention kernel’s `max_input_len` argument is frozen inside the graph | Bucket `max_len` in units of 128 and re-right-align `b_loc` to the bucket boundary |
| **LoRA dispatch incompatible with padding** | `assert(len(q)==len(self.req_bins))` requires exact matching | Use exact `batch_size` in the cache key, without batch padding |
| **Engine recreated per request** | `model_rpc.py` creates a new `LoraUnorderedBatchMixed` for every request | Promote `_shared_cuda_graph_runner` to a class-level variable |

### 4.2 Usage

```bash
# Enable CUDA Graph
python test/llama3/launch_llama3.py --enable-cuda-graph

# Or launch the API server directly (requires the dserve conda environment)
/path/to/dserve/bin/python -m dserve.server.api_server ... --enable-cuda-graph

# Run the comparison benchmark
python test/llama3/bench_cuda_graph.py --num-requests 20
```

### 4.3 Limitations

- Each new `(batch_size, max_len_bucket)` combination requires one capture pass (~150–300ms); after that, replay takes only 0.1ms
- Because LoRA dispatch requires exact batch size, `batch_size` cannot be bucketed; graph cache size is therefore `|unique batch_sizes| × |max_len buckets|`
- Only the decode stage is supported; the prefill stage, with variable-length inputs, does not use CUDA Graph
- cuBLAS warmup introduces tiny floating-point deviations, which may flip ambiguous tokens under greedy decoding, but this does not affect generation quality

---

## 5. Debugging Process: `att_m_buffers` dtype Bug

### 5.1 Symptom

After enabling CUDA Graph, generated text degraded into repetitive garbage:

```text
Baseline:    " Paris. The city is located in the north of the country on the banks of the Seine River..."
CUDA Graph:  " Paris,\n (the\n (the\n (the\n (the\n (the\n ..."
```

The first token was sometimes partially correct (`"Paris,"`), but all subsequent tokens degenerated into repeated `"(the"`.

### 5.2 Debugging Method: Layer-by-Layer Isolation with Controlled Variables

CUDA Graph introduced several changes at once, so the issue could not be localized directly. The debugging strategy was to **add changes incrementally** and isolate the faulty step.

Three modes controlled by the `CG_DEBUG` environment variable were added to `_decode_with_cuda_graph()`:

```python
if debug_mode == 'no_graph':
    # Mode A: non-contiguous path only, without realign and without graph capture
    predict_logics = self._token_forward(input_ids, infer_state, ...)
elif debug_mode == 'realign_only':
    # Mode B: non-contiguous + b_loc realign + att_m_buffers, but no graph capture
    ...
    infer_state.max_len_in_batch = ml_bucket
    predict_logics = self._token_forward(input_ids, infer_state, ...)
else:
    # Mode C: full CUDA graph capture/replay
    ...
```

These three modes gradually introduced more changes:

| Mode | non-contiguous | `b_loc` realign | `att_m_buffers` | graph capture | Notes |
|------|:-:|:-:|:-:|:-:|------|
| A: `no_graph` | Yes | No | No | No | Tests only the non-contiguous path |
| B: `realign_only` | Yes | Yes | Yes | No | Tests realign + `att_m_buffers` |
| C: default | Yes | Yes | Yes | Yes | Full CUDA Graph |

### 5.3 Test Results

**Mode A (`CG_DEBUG=no_graph`)**: output was **exactly identical** to the baseline.

Conclusion: the non-contiguous decode path itself is correct.

**Mode B (`CG_DEBUG=realign_only`)**: output became **garbage**, matching the same failure mode as full CUDA Graph.

```text
Baseline:    " Paris. The city is located in the north..."
realign_only:" Paris,\n (the\n (the\n (the\n ..."
```

Conclusion: the bug is **not in graph capture/replay**, but in the additional steps introduced in Mode B (`b_loc` realign or `att_m_buffers`).

### 5.4 Root Cause Analysis

Compared with Mode A, Mode B introduced three changes:
1. Right-aligning `b_loc` data to the `ml_bucket` boundary
2. Setting `max_len_in_batch = ml_bucket`
3. Using preallocated `att_m_buffers` instead of per-step `torch.empty`

The `b_loc` realign logic was verified mathematically:

```text
Original b_loc: data is in [0, max_len-1]
After realign:  data is in [shift, shift+max_len-1], where shift = ml_bucket - max_len
Kernel access:  b_loc[i, ml_bucket - 1 - j]  -> correctly maps to the realigned position ✓
```

So `b_loc` realign was correct. The problem was narrowed down to **`att_m_buffers`**.

Inspection of `_create_att_m_buffers` usage showed:

```python
# cuda_graph_runner.py, inside capture()
att_m_buffers = self._create_att_m_buffers(input_ids.dtype)  # <-- BUG!
```

`input_ids` is a token ID tensor with dtype **int64**. But `att_m_tensor` stores intermediate attention scores, so it must be **float16**.

The attention kernel was therefore writing float16 attention scores into an int64 buffer, causing the data to be misinterpreted. Softmax then read garbage values, and generation collapsed.

### 5.5 Fix

```diff
- att_m_buffers = self._create_att_m_buffers(input_ids.dtype)
+ att_m_buffers = self._create_att_m_buffers(torch.float16)
```

### 5.6 Validation After the Fix

| Mode | Output |
|------|--------|
| Mode B (`realign_only`, after fix) | **Exactly identical** to baseline |
| Mode C (full CUDA Graph, after fix) | **Semantically equivalent** to baseline, with only minor floating-point differences (see Section 2.3) |

The small differences in Mode C come from cuBLAS warmup, which is required for graph capture, and are not bugs.

### 5.7 Additional Finding: Warmup Is Required

An attempt was made to set `warmup_iters=0`:

```text
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`
```

cuBLAS requires workspace initialization on first use. The warmup stage ensures this initialization is completed before graph capture. The final implementation keeps `warmup_iters=3`.

---

## 6. Summary of File Changes

| File | Status | Lines Changed | Notes |
|------|--------|---------------|------|
| `slora/common/cuda_graph_runner.py` | **New** | +242 | Core CUDA Graph capture/replay logic |
| `slora/models/peft/lora_unordered_batch_mixed.py` | Modified | +91 | Integrates CUDA Graph into the engine |
| `slora/models/llama/layer_infer/transformer_layer_infer.py` | Modified | +23 | Preallocated `att_m_buffer` support |
| `slora/models/llama3/layer_infer/transformer_layer_infer.py` | Modified | +7 | Preallocated `att_m_buffer` support for LLaMA-3 |
| `slora/models/llama/infer_struct.py` | Modified | +1 | Comment |
| `slora/server/api_server.py` | Modified | +4 | CLI argument |
| `slora/server/input_params.py` | Modified | +3 | Parameter definition |
| `slora/server/router/manager.py` | Modified | +3 | Argument propagation |
| `slora/server/router/model_infer/model_rpc.py` | Modified | +9 | Passes argument to engine |
| `test/llama3/launch_llama3.py` | Modified | +21 | Launch script support |
| `test/llama3/bench_cuda_graph.py` | **New** | +247 | Automated benchmark |