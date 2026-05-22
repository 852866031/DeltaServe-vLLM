"""
Opt-in CUDA graph wrapper around LlamaSFTBackwardService's context backward.

Phase 3 scope: captures `_backprop_ffn` once per layer at a single fixed
size (`max_saved_finetuning_tokens` from InputParams.FinetuneParams). The
rest of `_context_backward` — loss, logit backward, post-layer backward,
attention backward, optimizer step — stays eager. Only the FFN backward,
which is launch-overhead-heavy (rmsnorm forward/backward + 6 gemms + a
handful of elementwise ops per layer), is replayed from a captured graph.

The size is fixed because the fine-tuning token budget per backward is
configured at system start. That gives us exactly one graph per layer —
no bucketing.

The runner never mutates the service's math. It only:
  1. stages inputs into persistent static buffers sized to S_max,
  2. replays a captured graph that wraps `svc._backprop_ffn(...)`, and
  3. hands back a view of the output to feed the eager attention backward.

Llama3 gets this for free: `Llama3SFTBackwardService` inherits `__init__`
and `start_service`, and the runner calls `svc._backpop_attention` and
`svc._backprop_ffn` polymorphically.
"""

from __future__ import annotations

import time
import traceback
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from dserve.models.llama.SFT_service import LlamaSFTBackwardService


def _graph_print(*args, sep: str = " ", end: str = "\n") -> None:
    color = "\033[35m"
    reset = "\033[0m"
    text = sep.join(str(a) for a in args)
    print(f"{color}[BWD-GRAPH]: {text}{reset}", end=end)


class GraphedBackwardRunner:
    """See module docstring."""

    # Print a GPU-event timing breakdown every Nth real backward. 0 = off.
    # Profiled calls pay one extra cudaEventSynchronize (a few hundred μs)
    # and allocate ~2*num_layers events, so this is cheap to leave on.
    PROFILE_EVERY: int = 10

    def __init__(self, service: "LlamaSFTBackwardService") -> None:
        self.svc = service

        self._graph_pool = None
        self._s_max: int = 0
        self._static_ffn_in: Optional[torch.Tensor] = None
        self._static_ffn_dy: Optional[torch.Tensor] = None
        self._static_ffn_out: Optional[torch.Tensor] = None
        self._ffn_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._ffn_failed: set = set()   # layer_id blacklist after capture failure

        # Padded-attention capture state. Only populated for Llama3 services
        # (which define _backpop_attention_padded_core and the ATTN_BN_MAX/
        # L_MAX constants) and only when USE_GRAPHED_ATTENTION is set on the
        # base class. Persistent ctx and per-layer fp32 LoRA grad buffers
        # live OUTSIDE the graph pool (regular caching allocator) — this is
        # what prevents the previous attempt's pool-aliasing NaN bug.
        self._attn_ctx: Optional[SimpleNamespace] = None
        self._persistent_lora_grads: Dict[int, torch.Tensor] = {}
        self._g_attn_padded: Dict[int, torch.cuda.CUDAGraph] = {}
        self._attn_failed: set = set()

        self._prepared = False
        self._disabled = False
        self._warming_up = False
        self._replay_count = 0
        self._fallback_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Allocate persistent input/output buffers AND eagerly capture every
        per-layer FFN backward graph up front so steady-state serving never
        pays the first-use capture tax. Called once after the service has
        finished its own init (so embed_dim_, model_weights, bwd_stream and
        shared_activations all exist)."""
        if self._prepared:
            return

        svc = self.svc
        # Single fixed size taken from input_params.finetuning_params; this
        # is the system-wide upper bound on the number of fine-tuning tokens
        # that can appear in one backward call.
        s_max = int(svc.max_saved_finetuning_tokens)
        D = int(svc.embed_dim_)
        device = torch.cuda.current_device()
        # Match the dtype of the shared activations so copy_() is a plain
        # memcpy (no implicit cast).
        dtype = svc.shared_activations.transformer_out_activations[0].dtype

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()

        # ----- persistent static I/O buffers (one set, sized to S_max) -----
        self._s_max = s_max
        self._static_ffn_in = torch.zeros((s_max, D), device=device, dtype=dtype)
        self._static_ffn_dy = torch.zeros((s_max, D), device=device, dtype=dtype)
        self._static_ffn_out = torch.zeros((s_max, D), device=device, dtype=dtype)
        self._graph_pool = torch.cuda.graph_pool_handle()

        torch.cuda.synchronize()
        mem_after_buffers = torch.cuda.memory_allocated()

        # ----- eager graph capture: one graph per layer -----
        num_layers = svc.num_layers
        capture_start = time.time()
        for layer_id in range(num_layers):
            self._ensure_ffn_graph(layer_id)
        capture_elapsed = time.time() - capture_start

        torch.cuda.synchronize()
        mem_after_capture = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()

        MB = 1024.0 * 1024.0
        buffer_mb = (mem_after_buffers - mem_before) / MB
        graph_mb = (mem_after_capture - mem_after_buffers) / MB
        total_mb = (mem_after_capture - mem_before) / MB
        reserved_mb = (reserved_after - reserved_before) / MB
        captured = len(self._ffn_graphs)
        failed = len(self._ffn_failed)

        _graph_print(
            f"Prepared. S_max={s_max}, D={D}, layers={num_layers}, dtype={dtype}"
        )
        _graph_print(
            f"Captured {captured}/{num_layers} FFN graphs in "
            f"{capture_elapsed:.2f}s ({failed} failed)"
        )
        _graph_print(
            f"Memory (FFN): static buffers {buffer_mb:.1f} MB + graph state "
            f"{graph_mb:.1f} MB = {total_mb:.1f} MB allocated "
            f"(reserved grew by {reserved_mb:.1f} MB)"
        )

        # ----- Padded attention capture (llama3 only) -----
        if (
            getattr(svc, "USE_GRAPHED_ATTENTION", False)
            and hasattr(svc, "_backpop_attention_padded_core")
            and hasattr(svc, "ATTN_BN_MAX")
            and hasattr(svc, "ATTN_L_MAX")
        ):
            self._capture_padded_attention(device)
        else:
            _graph_print(
                "Padded attention capture skipped "
                f"(USE_GRAPHED_ATTENTION={getattr(svc, 'USE_GRAPHED_ATTENTION', False)}, "
                f"has_core={hasattr(svc, '_backpop_attention_padded_core')})"
            )

        self._prepared = True

        # Warm up the full backward path (logit bwd + post-layer bwd +
        # graphed FFN replays + graphed attention replays) so cuBLAS
        # algorithm selection, Triton JIT and CUDA graph first-replay
        # setup all happen before the service reports READY.
        self._warmup_full_backward()

    def _capture_padded_attention(self, device) -> None:
        """Allocate persistent ctx + per-layer fp32 LoRA grad buffers
        OUTSIDE the graph pool, seed the ctx with a synthetic full-L_max
        request, and capture one CUDA graph per layer around
        svc._backpop_attention_padded_core(ctx, layer_weight, layer_id).

        All persistent buffers live in the default caching allocator, not
        the graph pool — this guarantees their addresses survive pool reuse
        by later graph replays, which was the root cause of the previous
        attempt's LoRA-grad NaN bug."""
        svc = self.svc
        Bn_max = int(svc.ATTN_BN_MAX)
        L_max = int(svc.ATTN_L_MAX)
        Hq = int(svc.num_heads_q_)
        Hkv = int(svc.num_heads_kv_)
        Hd = int(svc.head_dim_)
        D = int(svc.embed_dim_)
        s_max = self._s_max
        num_layers = svc.num_layers
        stream = svc.bwd_stream

        torch.cuda.synchronize()
        mem_before_attn = torch.cuda.memory_allocated()
        reserved_before_attn = torch.cuda.memory_reserved()

        # Persistent ctx — all fields fixed shape, all stable addresses.
        # Allocated OUTSIDE any torch.cuda.graph(...) context so they land
        # in the default caching allocator, not the graph pool.
        ctx = SimpleNamespace()
        ctx.flat_in_padded       = torch.zeros((s_max, D),                device=device, dtype=torch.float16)
        ctx.grad_flat_in_padded  = torch.zeros((s_max, D),                device=device, dtype=torch.float16)
        ctx.bn_idx               = torch.zeros((s_max,),                  device=device, dtype=torch.long)
        ctx.pos_idx              = torch.zeros((s_max,),                  device=device, dtype=torch.long)
        ctx.pad_x_prev           = torch.zeros((Bn_max, L_max, D),        device=device, dtype=torch.float16)
        ctx.pad_grad_out         = torch.zeros((Bn_max, L_max, D),        device=device, dtype=torch.float16)
        ctx.pad_cos              = torch.zeros((Bn_max, L_max, Hd // 2),  device=device, dtype=svc.model_weights._cos_cached.dtype)
        ctx.pad_sin              = torch.zeros((Bn_max, L_max, Hd // 2),  device=device, dtype=svc.model_weights._sin_cached.dtype)
        ctx.key_pad_mask         = torch.ones ((Bn_max, L_max),           device=device, dtype=torch.bool)
        ctx.causal_mask          = torch.triu(
            torch.ones((L_max, L_max), device=device, dtype=torch.bool),
            diagonal=1,
        )
        ctx.flat_grad_x_prev_out = torch.zeros((s_max, D),                device=device, dtype=torch.float16)
        self._attn_ctx = ctx

        # Per-layer persistent fp32 LoRA grad buffers — outside pool.
        for layer_id in range(num_layers):
            leaf = svc.adapter_weights.lora_weights[layer_id]
            self._persistent_lora_grads[layer_id] = torch.zeros_like(leaf, dtype=torch.float32)

        torch.cuda.synchronize()
        mem_after_attn_buffers = torch.cuda.memory_allocated()

        # Seed ctx with a synthetic full-length request so capture warmup
        # has sensible data in every field. Temporarily install a fake
        # Activations with seq_lens_py = [s_max] (single request filling
        # every slot). The service's _fill_attn_ctx_eager handles the rest.
        from dserve.models.llama.SFT_service import Activations
        saved_activations = svc.activations
        fake = Activations()
        # One synthetic request of length min(s_max, L_max) tokens — fits
        # the [S_max, D] input shape, builds valid bn_idx/pos_idx.
        # Clamp to L_max since per-request length must be <= L_max.
        fake_len = min(s_max, L_max)
        fake.seq_lens_py = [fake_len]
        fake.b_start_py = [0]
        svc.activations = fake

        synthetic_last_layer_input = torch.zeros((fake_len, D), device=device, dtype=torch.float16)
        synthetic_grad_ffn_input = torch.zeros((fake_len, D), device=device, dtype=torch.float16)

        try:
            svc._fill_attn_ctx_eager(ctx, synthetic_last_layer_input, synthetic_grad_ffn_input)

            capture_start_attn = time.time()
            for layer_id in range(num_layers):
                layer_weight = svc.model_weights.trans_layers_weight[layer_id]
                # Rebind lora_grad_out before each layer's capture so the
                # captured .copy_ kernel records a destination pointer into
                # THIS layer's persistent fp32 buffer. After capture,
                # overwriting ctx.lora_grad_out for the next layer does not
                # affect graphs already recorded.
                ctx.lora_grad_out = self._persistent_lora_grads[layer_id]

                try:
                    # Two eager warmup iters on bwd_stream to prime Triton /
                    # cuBLAS before capture.
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            svc._backpop_attention_padded_core(ctx, layer_weight, layer_id)
                    stream.synchronize()

                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=stream, pool=self._graph_pool):
                        svc._backpop_attention_padded_core(ctx, layer_weight, layer_id)
                    self._g_attn_padded[layer_id] = g

                except Exception as e:
                    _graph_print(f"Padded attention capture failed layer={layer_id}: {e}")
                    traceback.print_exc()
                    self._attn_failed.add(layer_id)

            capture_elapsed_attn = time.time() - capture_start_attn
        finally:
            svc.activations = saved_activations

        # Attach persistent fp32 grad buffers to the LoRA leaves so
        # optimizer.step() reads from stable-address persistent storage.
        # The runner's run() preamble re-attaches these every backward,
        # guarding against zero_grad(set_to_none=True) between calls.
        for layer_id in range(num_layers):
            svc.adapter_weights.lora_weights[layer_id].grad = self._persistent_lora_grads[layer_id]

        torch.cuda.synchronize()
        mem_after_attn_capture = torch.cuda.memory_allocated()
        reserved_after_attn = torch.cuda.memory_reserved()

        MB = 1024.0 * 1024.0
        attn_buffer_mb = (mem_after_attn_buffers - mem_before_attn) / MB
        attn_graph_mb = (mem_after_attn_capture - mem_after_attn_buffers) / MB
        attn_total_mb = (mem_after_attn_capture - mem_before_attn) / MB
        attn_reserved_mb = (reserved_after_attn - reserved_before_attn) / MB
        captured_attn = len(self._g_attn_padded)
        failed_attn = len(self._attn_failed)

        _graph_print(
            f"Captured {captured_attn}/{num_layers} padded attention graphs in "
            f"{capture_elapsed_attn:.2f}s ({failed_attn} failed). "
            f"Bn_max={Bn_max}, L_max={L_max}"
        )
        _graph_print(
            f"Memory (attn): persistent buffers {attn_buffer_mb:.1f} MB + graph state "
            f"{attn_graph_mb:.1f} MB = {attn_total_mb:.1f} MB allocated "
            f"(reserved grew by {attn_reserved_mb:.1f} MB)"
        )
        if failed_attn > 0:
            _graph_print(
                "Some attention captures failed — affected layers will "
                "fall through to eager _backpop_attention_padded."
            )

    def _warmup_full_backward(self) -> None:
        """Run one dummy backward through the graphed path with synthetic
        activations sized to S_max. Primes cuBLAS / Triton / driver caches
        without stepping the optimizer (so no weight drift)."""
        from dserve.models.llama.SFT_service import Activations

        svc = self.svc
        shared = svc.shared_activations
        s_max = self._s_max
        shared_cap = int(shared.transformer_out_activations[0].shape[0])
        if s_max > shared_cap:
            _graph_print(
                f"Warmup skipped: S_max={s_max} > shared activation cap {shared_cap}"
            )
            return
        if svc.finetuning_optimizer is None:
            _graph_print("Warmup skipped: optimizer not initialized yet")
            return

        saved_activations = svc.activations
        self._warming_up = True
        warmup_start = time.time()
        try:
            # One synthetic request with s_max saved tokens. Shapes mirror
            # what receive_requests_info builds at serve time.
            fake = Activations()
            fake.logit_list = [shared.logit_tensor[:s_max, :]]
            fake.concat_input_ids = shared.concat_input_ids[:s_max + 1]
            fake.transformer_out_activations = [
                shared.transformer_out_activations[i][:s_max]
                for i in range(svc.num_layers)
            ]
            fake.attention_out_activations = [
                shared.attention_out_activations[i][:s_max]
                for i in range(svc.num_layers)
            ]
            fake.input_layer_output = shared.input_layer_output[:s_max]
            svc.activations = fake

            with torch.cuda.stream(svc.bwd_stream):
                self.run()
            svc.bwd_stream.synchronize()
        except Exception as e:
            _graph_print(f"Warmup backward failed: {e}")
            traceback.print_exc()
        finally:
            svc.activations = saved_activations
            self._warming_up = False

        # Clear grads accumulated during warmup so the first real backward
        # starts with a clean slate (optimizer state is still lazily
        # allocated on first real step, which is a few ms — not the 100+ ms
        # we were chasing).
        svc.finetuning_optimizer.zero_grad(set_to_none=True)
        warmup_elapsed = time.time() - warmup_start
        _graph_print(f"Warmup backward completed in {warmup_elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture_ffn(self, layer_id: int) -> None:
        svc = self.svc
        layer_weight = svc.model_weights.trans_layers_weight[layer_id]

        in_buf = self._static_ffn_in
        dy_buf = self._static_ffn_dy
        out_buf = self._static_ffn_out

        stream = svc.bwd_stream

        # Triton kernels used by rmsnorm_forward/backward compile on first
        # call — run two warmup iters on the same stream before capture so
        # the compile happens outside the graph region.
        with torch.cuda.stream(stream):
            for _ in range(2):
                tmp = svc._backprop_ffn(in_buf, dy_buf, layer_weight)
                out_buf.copy_(tmp)
        stream.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream, pool=self._graph_pool):
            tmp = svc._backprop_ffn(in_buf, dy_buf, layer_weight)
            out_buf.copy_(tmp)

        self._ffn_graphs[layer_id] = g

    def _ensure_ffn_graph(self, layer_id: int) -> bool:
        """Capture on first use. Returns True if the graph is usable."""
        if layer_id in self._ffn_graphs:
            return True
        if layer_id in self._ffn_failed:
            return False
        try:
            self._capture_ffn(layer_id)
            return True
        except Exception as e:
            _graph_print(f"FFN capture failed for layer={layer_id}: {e}")
            traceback.print_exc()
            self._ffn_failed.add(layer_id)
            return False

    # ------------------------------------------------------------------
    # Per-layer replay
    # ------------------------------------------------------------------

    def _ffn_replay(
        self,
        layer_id: int,
        ffn_input: torch.Tensor,
        output_grad: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Stage inputs, replay, return a view of the valid output rows.
        Returns None if the graph isn't usable — caller falls back to eager
        for this layer."""
        s_actual = int(ffn_input.shape[0])
        in_buf = self._static_ffn_in
        dy_buf = self._static_ffn_dy
        out_buf = self._static_ffn_out

        if s_actual < self._s_max:
            in_buf[s_actual:].zero_()
            dy_buf[s_actual:].zero_()
        in_buf[:s_actual].copy_(ffn_input)
        dy_buf[:s_actual].copy_(output_grad)

        if not self._ensure_ffn_graph(layer_id):
            return None

        self._ffn_graphs[layer_id].replay()
        return out_buf[:s_actual]

    def _attn_padded_replay(
        self,
        last_layer_input: torch.Tensor,
        grad_ffn_input: torch.Tensor,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        """Per-layer eager prep → replay captured core → eager post.

        The per-BACKWARD prep (scatter indices, rotary tables, pad mask)
        is done once by the runner's `run()` preamble via
        `svc._fill_attn_ctx_per_backward(ctx)`. This method only does the
        per-LAYER update (copying this layer's flat inputs into the
        persistent padded buffers) and then replays.

        Returns None if the captured graph isn't available for this layer;
        the caller falls back to the eager padded wrapper."""
        if layer_id not in self._g_attn_padded:
            return None

        svc = self.svc
        ctx = self._attn_ctx

        # Per-layer update only. Everything else in ctx was set by the
        # per-backward prep earlier in run().
        svc._fill_attn_ctx_per_layer(ctx, last_layer_input, grad_ffn_input)

        # Replay the captured shape-stable core for this layer.
        self._g_attn_padded[layer_id].replay()

        # The captured gather already wrote ctx.flat_grad_x_prev_out
        # (persistent buffer, safe address). Slice down to the real valid
        # prefix and hand it back.
        s_actual = int(last_layer_input.shape[0])
        return ctx.flat_grad_x_prev_out[:s_actual]

    def _graphed_lora_context_backward(
        self,
        layer_id: int,
        output_grad: torch.Tensor,
        ffn_done_evt: Optional["torch.cuda.Event"] = None,
        attn_done_evt: Optional["torch.cuda.Event"] = None,
    ) -> torch.Tensor:
        """Mirror of svc._lora_context_backward with the FFN step replayed,
        and (when available + the batch fits the padded budget) the
        attention step replayed via a captured shape-stable core. Falls
        back to the eager padded wrapper or the monolithic attention bwd
        when the captured path isn't usable.

        Optional cuda events may be passed in to mark the end of the FFN
        and attention sub-regions for profiling."""
        svc = self.svc
        layer_weight = svc.model_weights.trans_layers_weight[layer_id]
        ffn_input = svc.activations.attention_out_activations[layer_id]

        grad_ffn_input = self._ffn_replay(layer_id, ffn_input, output_grad)
        if grad_ffn_input is None:
            # Single-layer fallback: recompute FFN bwd eagerly.
            grad_ffn_input = svc._backprop_ffn(ffn_input, output_grad, layer_weight)

        if ffn_done_evt is not None:
            ffn_done_evt.record()

        if layer_id == 0:
            last_layer_input = svc.activations.input_layer_output
        else:
            last_layer_input = svc.activations.transformer_out_activations[layer_id - 1]

        svc._maybe_pause()

        # Attention dispatch:
        #   1. If capture is enabled and the batch fits the padded budget →
        #      graph replay the padded core.
        #   2. Else if capture is enabled but the batch overflows the budget
        #      → monolithic _backpop_attention directly (graceful fallback
        #      for oversized tail batches, no crash).
        #   3. Else (capture not enabled / not captured) → eager padded
        #      wrapper if available (which itself falls back to monolithic
        #      on overflow), or monolithic directly.
        result = None
        padded_available = (
            getattr(svc, "USE_GRAPHED_ATTENTION", False)
            and hasattr(svc, "_backpop_attention_padded")
        )
        if padded_available:
            seq_lens_py = svc.activations.seq_lens_py
            batch_fits = (
                len(seq_lens_py) <= svc.ATTN_BN_MAX
                and (len(seq_lens_py) == 0 or max(seq_lens_py) <= svc.ATTN_L_MAX)
            )
            if batch_fits and layer_id in self._g_attn_padded:
                result = self._attn_padded_replay(last_layer_input, grad_ffn_input, layer_id)
            elif batch_fits:
                # Captured graph missing for this layer but batch fits —
                # use the eager padded wrapper (same math as the graphed
                # path, just without the replay speedup).
                result = svc._backpop_attention_padded(
                    last_layer_input, grad_ffn_input, layer_weight, layer_id
                )
            # If not batch_fits, fall through to monolithic below.

        if result is None:
            result = svc._backpop_attention(
                last_layer_input, grad_ffn_input, layer_weight, layer_id
            )

        if attn_done_evt is not None:
            attn_done_evt.record()
        return result

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> Tuple[bool, torch.Tensor, int]:
        """Graphed replacement for svc._context_backward. On any error, the
        caller catches and falls back to the eager path for the rest of the
        process."""
        svc = self.svc

        # Re-attach persistent fp32 LoRA grad buffers. zero_grad(set_to_none=True)
        # between backwards nulls the leaf's .grad; without this rebind the
        # captured .copy_ writes into the persistent buffer as usual but
        # optimizer.step() sees .grad is None and skips the param.
        for layer_id, pg in self._persistent_lora_grads.items():
            svc.adapter_weights.lora_weights[layer_id].grad = pg

        logits_and_targets, total_tokens_to_process = svc.get_logits_and_targets()
        loss = svc.compute_total_loss(logits_and_targets)

        # Per-backward attention prep: builds scatter indices, pad_cos/sin,
        # and key_pad_mask once — all fields that depend on seq_lens_py but
        # not on any per-layer input. The per-layer path (_attn_padded_replay)
        # then only has to copy this layer's flat inputs into the persistent
        # buffers, avoiding 32× redundant work.
        if (
            self._attn_ctx is not None
            and len(self._g_attn_padded) > 0
            and getattr(svc, "USE_GRAPHED_ATTENTION", False)
        ):
            seq_lens_py = svc.activations.seq_lens_py
            if (
                len(seq_lens_py) <= svc.ATTN_BN_MAX
                and (len(seq_lens_py) == 0 or max(seq_lens_py) <= svc.ATTN_L_MAX)
            ):
                svc._fill_attn_ctx_per_backward(self._attn_ctx)

        s_total = int(svc.activations.transformer_out_activations[-1].shape[0])
        if s_total > self._s_max:
            # Invariant violation: activations exceed the configured cap.
            # Should not happen, but we fall through to eager for safety.
            self._fallback_count += 1
            _graph_print(
                f"s_total={s_total} > S_max {self._s_max}; falling through to eager "
                f"for this call (replays={self._replay_count}, fallbacks={self._fallback_count})"
            )
            logit_grad = svc._logit_backward(logits_and_targets)
            grad_transformer_out = svc._post_layer_backward(logit_grad, svc.model_weights.pre_post_weight)
            for i in reversed(range(svc.num_layers)):
                svc._maybe_pause()
                grad_transformer_out = svc._lora_context_backward(i, grad_transformer_out)
            return True, loss, total_tokens_to_process

        if not self._warming_up and False:
            _graph_print(
                f"Using CUDA graph size={self._s_max} for s_total={s_total}"
            )

        # Decide whether to time this call. Skip during warmup so the warmup
        # backward doesn't skew the first printed breakdown.
        do_profile = (
            not self._warming_up
            and self.PROFILE_EVERY > 0
            and (self._replay_count % self.PROFILE_EVERY == 0)
            and False
        )

        if do_profile:
            num_layers = svc.num_layers
            e_start = torch.cuda.Event(enable_timing=True)
            e_after_lp = torch.cuda.Event(enable_timing=True)
            e_end = torch.cuda.Event(enable_timing=True)
            e_ffn_done = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
            e_attn_done = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
            e_start.record()
        else:
            e_start = e_after_lp = e_end = None
            e_ffn_done = e_attn_done = None

        logit_grad = svc._logit_backward(logits_and_targets)
        grad_transformer_out = svc._post_layer_backward(logit_grad, svc.model_weights.pre_post_weight)

        if do_profile:
            e_after_lp.record()

        for k, i in enumerate(reversed(range(svc.num_layers))):
            svc._maybe_pause()
            grad_transformer_out = self._graphed_lora_context_backward(
                i,
                grad_transformer_out,
                ffn_done_evt=(e_ffn_done[k] if do_profile else None),
                attn_done_evt=(e_attn_done[k] if do_profile else None),
            )

        if do_profile:
            e_end.record()
            # One sync to make every recorded event readable.
            torch.cuda.synchronize()
            logit_post_ms = e_start.elapsed_time(e_after_lp)
            total_ms = e_start.elapsed_time(e_end)
            ffn_total_ms = 0.0
            attn_total_ms = 0.0
            prev_evt = e_after_lp
            for k in range(svc.num_layers):
                ffn_total_ms += prev_evt.elapsed_time(e_ffn_done[k])
                attn_total_ms += e_ffn_done[k].elapsed_time(e_attn_done[k])
                prev_evt = e_attn_done[k]
            other_ms = max(0.0, total_ms - logit_post_ms - ffn_total_ms - attn_total_ms)
            _graph_print(
                f"[PROFILE] logit+post {logit_post_ms:6.2f} ms | "
                f"ffn graph {ffn_total_ms:6.2f} ms | "
                f"attn eager {attn_total_ms:6.2f} ms | "
                f"other {other_ms:5.2f} ms | "
                f"total {total_ms:6.2f} ms "
                f"(s_total={s_total}, layers={svc.num_layers})"
            )

        self._replay_count += 1
        return True, loss, total_tokens_to_process
