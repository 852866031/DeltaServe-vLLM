# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``DeltaServeWorker``: vLLM's GPU ``Worker`` plus the co-serving setup.

Selected through vLLM's own ``parallel_config.worker_cls`` whenever finetuning
is enabled (``config/vllm.py``), so the upstream worker carries no DeltaServe
code. On top of the stock worker it

- spawns this rank's backward (SFT) child once the device is up — the analogue
  of DeltaServe's ``model_rpc.py:150-178`` (under TP: one child per rank, each
  pinned to its own GPU);
- after the weights load, shares the base weights + the fp32 FT-adapter masters
  with the child (CUDA IPC), allocates the FT activation buffers + their hooks,
  and shares vLLM's served LoRA buffers so the trainer publishes into them;
- answers the ``deltaserve_{start,stop}_finetuning`` collective RPCs.
"""

import os

import torch

from vllm.v1.worker.gpu_worker import Worker


class DeltaServeWorker(Worker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.parallel_config.distributed_executor_backend == "mp":
            # A worker process of its own (TP): arm before the model load, so a
            # crashed EngineCore cannot leak it (and its share of the model) —
            # nor, through it, the backward child. Under the uniproc executor
            # this is the EngineCore process itself: left alone.
            from vllm.deltaserve.backward_process import arm_parent_death_signal

            arm_parent_death_signal(f"worker rank={self.rank}")

    def init_device(self):
        super().init_device()
        from vllm.deltaserve import dprint
        from vllm.deltaserve.backward_process import BackwardProcess
        from vllm.deltaserve.model_runner import DeltaServeModelRunner

        DeltaServeModelRunner.adopt(self.model_runner)

        ft_cfg = self.vllm_config.finetune_config
        arch = self.model_config.hf_config.architectures[0]
        tp_size = self.parallel_config.tensor_parallel_size
        dprint(f"Worker rank={self.rank} local_rank={self.local_rank} "
               f"tp_size={tp_size}: spawning backward SFT child on "
               f"cuda:{self.local_rank} (arch={arch})")
        self.backward_process = BackwardProcess(
            mps_percentage=ft_cfg.backward_mps_percentage,
            device_index=self.local_rank,
            service_name=arch,
        )
        self.backward_process.start()
        dprint(f"Worker rank={self.rank} local_rank={self.local_rank}: backward "
               f"child pid={getattr(self.backward_process._proc, 'pid', None)} ready")

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        # Base weights are on the GPU now: share them (frozen, zero-copy) plus
        # the fp32 FT-adapter master weights with the backward child.
        self._maybe_share_finetuning_weights()
        # Activation buffers + accumulation hooks, shared with the child and
        # handed to the model runner.
        self._maybe_setup_finetuning_accumulator()
        # Pin the FT adapter into a stable served slot and share vLLM's served
        # LoRA buffers, so the trainer publishes straight into what inference
        # reads (no worker-side refresh).
        self._maybe_share_ft_served_lora()

    def _maybe_share_finetuning_weights(self) -> None:
        """Share base + FT-adapter weights with the backward process (Step 3)."""
        backward_process = getattr(self, "backward_process", None)
        if backward_process is None:
            return

        import os

        from safetensors.torch import load_file

        from vllm.deltaserve import dprint

        ft_cfg = self.vllm_config.finetune_config

        # Base weights: frozen references, shared zero-copy (no detach copy).
        # Under TP>1 these are this rank's SHARDS (qkv/gate_up column-parallel,
        # o/down row-parallel, embed/lm_head vocab-parallel). The backward
        # service slices them by LOCAL dims (see _build_state); the meta below
        # carries tp_size/tp_rank so it knows the shard geometry.
        model = self.get_model()
        base_state = {name: p.detach() for name, p in model.named_parameters()}

        # [DeltaServe] Phase 7 / M2: TP shard geometry. tp_size=1 (uniproc) →
        # everything below is a no-op and the shared state is bit-identical to
        # the single-GPU path.
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        tp_size = int(get_tensor_model_parallel_world_size())
        tp_rank = int(get_tensor_model_parallel_rank())

        # FT adapter: load its weights as an fp32 master copy on this device.
        ft_state: dict[str, torch.Tensor] = {}
        path = ft_cfg.finetuning_lora_path
        if path:
            safetensors_path = os.path.join(path, "adapter_model.safetensors")
            if os.path.isfile(safetensors_path):
                raw = load_file(safetensors_path, device=str(self.device))
                ft_state = {k: v.float() for k, v in raw.items()}
                dprint(
                    f"loaded FT adapter fp32 from {safetensors_path}: "
                    f"{len(ft_state)} tensors"
                )
            else:
                dprint(
                    f"FT adapter safetensors not found at {safetensors_path}; "
                    f"sharing base weights only"
                )

        # Resolve the LM-head weight + trim/scale info the backward process needs
        # to reconstruct logits. Models that tie word embeddings (Qwen3-0.6B) have
        # the LM-head weight under the embedding param name; fall back to that
        # when there is no separate "lm_head.weight".
        hf_config = self.model_config.hf_config
        # With enable_lora vLLM wraps LoRA-capable modules, so a parameter can
        # be named "...embed_tokens.base_layer.weight" (Qwen3-0.6B ties lm_head
        # to the embedding and the wrapped embedding is what we see). Resolve
        # on the NORMALIZED names the child looks up (it strips ".base_layer."
        # the same way) and keep the raw name for our own indexing.
        _norm2raw = {k.replace(".base_layer.", "."): k for k in base_state}
        if "lm_head.weight" in _norm2raw:
            lm_head_key = "lm_head.weight"
        else:
            lm_head_key = next(
                (k for k in _norm2raw if k.endswith("embed_tokens.weight")), None)
        embed_weight_key = next(
            (k for k in _norm2raw if k.endswith("embed_tokens.weight")), None)
        lm_head_raw = _norm2raw.get(lm_head_key) if lm_head_key else None
        if lm_head_key is None:
            dprint("[deltaserve] WARNING: no lm_head / embed_tokens weight among the "
                   f"shared base tensors (e.g. {list(base_state)[:3]}) — the backward "
                   "child cannot build its state")

        # [DeltaServe] Phase 7 / M2: the LM head is vocab-parallel under TP, so
        # this rank only holds a vocab slice. head_backward needs the FULL logits
        # matrix. All-gather the shard over vLLM's TP group (both ranks call this
        # collectively during load_model) and hand the backward the full lm_head
        # — that keeps head_backward (and final-norm) UNCHANGED and removes
        # vocab-parallelism from the backward entirely. Llama-3's vocab (128256)
        # has no padding, so the gather is exactly [vocab_size, hidden]; trim
        # defensively for models vLLM pads.
        if tp_size > 1 and lm_head_raw is not None:
            shard = base_state[lm_head_raw]
            full_lm = get_tp_group().all_gather(shard.contiguous(), dim=0)
            full_lm = full_lm[: int(hf_config.vocab_size)].contiguous()
            base_state[lm_head_raw] = full_lm
            dprint(
                f"[tp] all-gathered lm_head '{lm_head_key}' shard "
                f"{tuple(shard.shape)} -> {tuple(full_lm.shape)} "
                f"(tp_size={tp_size}, tp_rank={tp_rank})"
            )

        from vllm.deltaserve.ft_meta import (
            build_backward_meta,
            read_lora_scaling,
        )

        # Everything the child needs to know about the model + this rank's
        # shard, built in one deltaserve-owned place (no family branches here).
        meta = build_backward_meta(
            hf_config, ft_cfg,
            lm_head_key=lm_head_key,
            embed_weight_key=embed_weight_key,
            has_final_norm="model.norm.weight" in _norm2raw,
            lora_scaling=read_lora_scaling(path),
            tp_size=tp_size, tp_rank=tp_rank)

        dprint(
            f"sharing weights with backward process: base={len(base_state)} "
            f"tensors, ft={len(ft_state)} tensors, lm_head_key={lm_head_key}"
        )
        summary = backward_process.share_weights(
            base_state, ft_state, print_hash=ft_cfg.print_weight_hash, meta=meta)

        # Optionally hash the first/last FFN weight (base) + first/last q_proj
        # LoRA-A (adapter) on the PARENT side and compare to the child's hashes.
        if ft_cfg.print_weight_hash:
            from vllm.deltaserve.backward_process import (
                print_hash_report,
                weight_hash_report,
            )

            parent_report = weight_hash_report(base_state, ft_state)
            print_hash_report(parent_report, "parent")
            child_report = summary.get("hash_report", {})
            match = parent_report == child_report
            dprint(
                f"weight hashes parent==child: {match} "
                f"(base/ft summary: {summary.get('base_numel')} / "
                f"{summary.get('ft_numel')} elems)"
            )

    def _maybe_setup_finetuning_accumulator(self) -> None:
        """Allocate FT activation buffers + hooks, share with backward (M2)."""
        backward_process = getattr(self, "backward_process", None)
        if backward_process is None:
            return

        from vllm.deltaserve import dprint
        from vllm.deltaserve.accumulate import FinetuneAccumulator
        from vllm.deltaserve.ft_meta import (
            effective_save_attn_qkv,
            saved_qkv_pre_transform,
        )

        arch = (self.model_config.hf_config.architectures or [None])[0]

        ft_cfg = self.vllm_config.finetune_config
        model = self.get_model()
        hf_config = self.model_config.hf_config
        hidden_size = hf_config.hidden_size
        dtype = next(model.parameters()).dtype

        # Derive q_size / kv_size for the optional post-RoPE attn-qkv save —
        # the hook target's q/k/v args are flat [n_total, q_size] /
        # [n_total, kv_size] so we need these widths to allocate the buffers.
        # Same formulas the backward service uses to slice qkv_proj
        # (gpu_worker meta + bwd_services/llama3._build_state).
        num_heads = int(hf_config.num_attention_heads)
        head_dim = int(getattr(hf_config, "head_dim", None)
                       or hidden_size // num_heads)
        num_kv_heads = int(getattr(hf_config, "num_key_value_heads", num_heads))

        # [DeltaServe] Phase 7 / M2: the mlp_gate_up / attn_qkv / attn_ctx hooks
        # capture the TP-SHARDED module outputs, so their buffer widths must be
        # LOCAL (per-rank). q/k/v and intermediate all shard by tp_size; the
        # residual-stream buffers (layer_in/final_*) stay full hidden_size.
        # tp_size=1 → local == full → single-GPU sizing unchanged.
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = int(get_tensor_model_parallel_world_size())
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0, (
            f"TP finetuning needs num_heads ({num_heads}) and num_kv_heads "
            f"({num_kv_heads}) divisible by tp_size ({tp_size})")
        q_size = (num_heads // tp_size) * head_dim          # local
        kv_size = (num_kv_heads // tp_size) * head_dim       # local
        inter_full = getattr(hf_config, "intermediate_size", None)
        local_inter = (int(inter_full) // tp_size) if inter_full else None
        if inter_full is not None:
            assert int(inter_full) % tp_size == 0, (
                f"TP finetuning needs intermediate_size ({inter_full}) "
                f"divisible by tp_size ({tp_size})")

        accumulator = FinetuneAccumulator(
            model=model,
            max_saved=ft_cfg.max_saved_finetuning_tokens,
            hidden_size=hidden_size,
            device=self.device,
            dtype=dtype,
            intermediate_size=local_inter,
            q_size=q_size,
            kv_size=kv_size,
            save_attn_qkv=effective_save_attn_qkv(arch, ft_cfg),
            attn_qkv_pre_transform=saved_qkv_pre_transform(arch),
            save_attn_ctx=bool(ft_cfg.save_attn_ctx),
            save_resid_mid=bool(ft_cfg.save_resid_mid),
            # Only the print_activation_hash check reads the post-norm output.
            save_final_hidden=bool(ft_cfg.print_activation_hash),
        )
        accumulator.register_hooks()
        # [mixed-fwd-cuda-graph] mark the model's modules for the graph-
        # capturable save op (before the first torch.compile trace, which
        # happens at the profile run — the hooks must see the armed op).
        if ft_cfg.graph_ft_batches:
            accumulator.install_save_points(model)
        ack = backward_process.share_activations(
            accumulator.buffers, print_hash=ft_cfg.print_activation_hash)
        dprint(f"shared activation buffers with backward process: {ack}")

        # Co-serving coordinator (process-wide singleton). The worker's load_model
        # runs before the scheduler is constructed, so we CREATE it here; the
        # FinetuneScheduler reuses the same instance (and sets min_sample_len).
        from vllm.deltaserve.coordinator import get_coordinator

        cap = int(ft_cfg.max_saved_finetuning_tokens)
        # per-step budget = full capacity (max_saved_finetuning_tokens): an
        # idle/FT-only step can fill the buffer in one eager forward. During
        # co-serving the SLO gate caps FT tokens once the estimator is warm.
        coordinator = get_coordinator(capacity=cap, per_step_budget=cap)
        coordinator.backward_process = backward_process
        # [DeltaServe] Phase 7 / M4.1: under TP>1 this worker-side coordinator is
        # a DIFFERENT process from the EngineCore scheduler. Relay mode makes it
        # a thin executor: the runner calls execute_trigger (on a relayed
        # SchedulerOutput command) + poll_backward_relay, and reports both back
        # via ModelRunnerOutput; the scheduler owns the trigger decision + store.
        coordinator.relay_mode = tp_size > 1
        # NOTE (known issue): under TP this worker-side coordinator keeps the
        # default ft_started=True and the scheduler mirrors it on the first
        # step, so finetune.start_on_launch=false does not hold admission
        # closed. Initialising the flag from start_on_launch here is not
        # enough: POST /start_finetuning only reaches the workers, and an idle
        # engine produces no ModelRunnerOutput to relay the flag back, so
        # fine-tuning would never start. A fix needs an EngineCore utility that
        # flips the scheduler's coordinator directly.
        # [forward_interruptible / tier C] Wire the accumulator's per-layer
        # abort check to the coordinator's threading.Event. The hook does a
        # cheap None-check first, so this is zero-cost when the feature is
        # off (event simply never gets set by the input thread).
        if ft_cfg.forward_interruptible:
            accumulator._abort_event = coordinator.ft_abort_event
            _shm_name = os.environ.get("DSERVE_FT_ARRIVAL_SHM")
            if tp_size > 1 and _shm_name:
                # [tier C under TP] the engine's arrival counter + the TP
                # group's gloo cpu_group make the abort decision rank-
                # symmetric (one CPU collective per layer boundary).
                from vllm.deltaserve.coordinator import (
                    FtAbortPoller,
                    FtArrivalSignal,
                )
                from vllm.distributed.parallel_state import get_tp_group

                _poller = FtAbortPoller(FtArrivalSignal.attach(_shm_name),
                                        get_tp_group().cpu_group)
                coordinator.ft_abort_poller = _poller
                accumulator._abort_poll = _poller.hook_check
                dprint("[deltaserve] forward_interruptible tier C under TP armed "
                       f"(arrival signal {_shm_name}, rank {get_tp_group().rank_in_group})")
            elif tp_size > 1:
                dprint("[deltaserve] forward_interruptible under TP: no arrival "
                       "signal from the engine — tier C inert (tiers A/B active)")
            else:
                from vllm.deltaserve.coordinator import LocalAbortPoller

                _poller = LocalAbortPoller(coordinator.ft_abort_event)
                coordinator.ft_abort_poller = _poller
                accumulator._abort_poll = _poller.hook_check

        # Inject into the runner: it sets the per-step mask + drives accumulation,
        # the accumulating offset, and the one-shot hash verification.
        self.model_runner._finetune_accumulator = accumulator
        self.model_runner._backward_process = backward_process
        self.model_runner._ft_coordinator = coordinator
        self.model_runner._ft_accum_verified = False

    def _maybe_share_ft_served_lora(self) -> None:
        """Pre-load + pin the FT LoRA adapter and share vLLM's served LoRA stacked
        buffers (CUDA-IPC) with the backward process so the trainer publishes updated
        weights straight into the tensors inference reads."""
        import re

        backward_process = getattr(self, "backward_process", None)
        if backward_process is None:
            return

        from vllm.deltaserve import dprint
        from vllm.deltaserve.ft_injector import FT_LORA_INT_ID
        from vllm.lora.request import LoRARequest

        ft_cfg = self.vllm_config.finetune_config
        path = ft_cfg.finetuning_lora_path
        if not path:
            return

        # Pre-load + pin the FT adapter so its served slot is allocated + stable
        # (no LRU eviction → the shared buffer rows stay ours).
        ft_lora = LoRARequest(lora_name="__finetune__",
                              lora_int_id=FT_LORA_INT_ID, lora_path=path)
        try:
            self.model_runner.add_lora(ft_lora)
            self.model_runner.pin_lora(FT_LORA_INT_ID)
        except Exception as e:  # noqa: BLE001
            dprint(f"[deltaserve] could not pre-load/pin FT adapter: {e}; "
                   f"served-weight publish disabled")
            return

        am = self.model_runner.lora_manager._adapter_manager
        slot = am.lora_index_to_id.index(FT_LORA_INT_ID)

        # Gather the FT adapter's served stacked buffers: qkv_proj (3 slices q/k/v)
        # + o_proj (1 slice), per layer. The trainer writes [slot] rows into these.
        layers: dict[int, dict] = {}
        for name, module in am.modules.items():
            m = re.search(r"layers\.(\d+)\.self_attn\.(qkv_proj|o_proj)$", name)
            if m is None:
                continue
            i = int(m.group(1))
            d = layers.setdefault(i, {})
            if m.group(2) == "qkv_proj":
                for s, proj in enumerate(("q", "k", "v")):
                    d[proj] = {"a": module.lora_a_stacked[s],
                               "b": module.lora_b_stacked[s]}
            else:
                d["o"] = {"a": module.lora_a_stacked[0],
                          "b": module.lora_b_stacked[0]}

        payload = {"slot": int(slot), "layers": layers}
        ack = backward_process.share_lora_buffers(payload)
        dprint(f"shared served LoRA buffers (slot={slot}, {len(layers)} layers) "
               f"with backward process: {ack}")

    def deltaserve_start_finetuning(self) -> bool:
        """[DeltaServe] Open FT admission. Invoked via collective_rpc from the
        POST /start_finetuning HTTP route. The coordinator is a process-wide
        singleton shared with the scheduler (single-GPU / uniproc), so flipping
        it here is visible to the scheduler's admission gate."""
        from vllm.deltaserve.coordinator import get_coordinator
        coord = get_coordinator()
        if coord is None:
            return False
        coord.start_finetuning()
        return True

    def deltaserve_stop_finetuning(self) -> bool:
        """[DeltaServe] Close FT admission. Invoked via collective_rpc from the
        POST /stop_finetuning HTTP route. Mirror of
        ``deltaserve_start_finetuning``. Idempotent: a stop on an already-
        stopped session is a no-op. After a stop, a follow-up start resumes
        from the same buffer / counter state — see ``Coordinator.stop_finetuning``
        for the preservation contract."""
        from vllm.deltaserve.coordinator import get_coordinator
        coord = get_coordinator()
        if coord is None:
            return False
        coord.stop_finetuning()
        return True
