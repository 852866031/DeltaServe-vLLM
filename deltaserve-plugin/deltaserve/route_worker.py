# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The route encoder inside a vLLM worker.

A routed request is encoded by a step the scheduler issues (the *encode step*):
``deltaserve_route_encode`` on every rank, between two model steps. Living in
the worker means the encode runs in the model's own CUDA context — nothing else
is on the GPU while it runs, so it costs exactly its own time (about
3.5 ms + 0.014 ms per token of the padded batch on an RTX 5090, fp32), and its
memory is part of what vLLM measures before it sizes the KV cache.

Under tensor parallelism one rank holds the encoder (``encoder_rank``); the
other ranks answer None at once. Every rank clears its backward child's grant
for the encode, exactly as for an inference prefill: the model step that
follows decides when the child gets the GPU back
(``DeltaServeModelRunner._ft_before_step``).

``RouteEncodeMixin`` goes in front of the worker class; ``RouteWorker`` is the
stock worker with it, for a server with adapter routing on and finetuning off.

Outside ``deltaserve.adapter_routing`` on purpose (like ``route_middleware``):
``DeltaServeWorker`` carries the mixin whether or not adapter routing is on,
and with it off nothing of that package is imported.
"""

import numpy as np

from vllm.v1.worker.gpu_worker import Worker


def encoder_rank(rc, tp_size: int) -> int:
    """The TP rank that holds the encoder: ``router_gpu`` when it names one of
    the model's cards, else the last rank."""
    gpu = getattr(rc, "router_gpu", None)
    if gpu is not None and 0 <= int(gpu) < tp_size:
        return int(gpu)
    return max(0, int(tp_size) - 1)


def pad_batch(batch: list, pad_id: int):
    """``(ids, mask)`` int64 arrays [n, longest] for token-id lists."""
    longest = max(len(ids) for ids in batch)
    ids = np.full((len(batch), longest), int(pad_id), np.int64)
    mask = np.zeros((len(batch), longest), np.int64)
    for i, row in enumerate(batch):
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1
    return ids, mask


class RouteEncodeMixin:
    _route_encoder = None
    _route_device = None
    _route_pad = 0

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        # Before vLLM measures what is left for the KV cache.
        self._load_route_encoder()

    def _route_tp_rank(self) -> int:
        if self.parallel_config.tensor_parallel_size <= 1:
            return 0
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())

    def _load_route_encoder(self) -> None:
        rc = self.vllm_config.adapter_routing_config
        if not rc.enable_adapter_routing:
            return
        from deltaserve import dprint

        tp = self.parallel_config.tensor_parallel_size
        if self._route_tp_rank() != encoder_rank(rc, tp):
            return
        import torch
        from transformers import AutoModel

        on_cpu = rc.router_gpu is None and str(rc.router_device).lower() == "cpu"
        device = "cpu" if on_cpu else str(self.device)
        model = AutoModel.from_pretrained(rc.encoder).to(device).eval()
        self._route_encoder, self._route_device = model, device
        self._route_pad = int(getattr(model.config, "pad_token_id", 0) or 0)
        # The largest batch an encode step may carry, once: its activations are
        # then in the allocator's pool when the KV cache is sized.
        length = int(rc.encoder_max_tokens)
        rows = max(1, int(rc.route_encode_max_tokens) // length)
        self._route_forward([[1] * length] * rows)
        if device != "cpu":
            torch.cuda.synchronize()
        if on_cpu:
            dprint("[adapter-routing] WARNING: the route encoder is on the CPU "
                   "(adapter_routing.router_device: cpu) — every routed request "
                   "waits for a CPU encode inside the engine step")
        dprint(f"[adapter-routing] route encoder {rc.encoder} loaded in worker "
               f"rank {self._route_tp_rank()} on {device}")

    def _route_forward(self, batch: list) -> np.ndarray:
        import torch

        ids, mask = pad_batch(batch, self._route_pad)
        with torch.inference_mode():
            ids_t = torch.from_numpy(ids).to(self._route_device)
            mask_t = torch.from_numpy(mask).to(self._route_device)
            hidden = self._route_encoder(
                input_ids=ids_t, attention_mask=mask_t).last_hidden_state
            m = mask_t.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            return pooled.cpu().numpy()

    # ----------------------------------------------------------- cold start
    def execute_model(self, scheduler_output, *args, **kwargs):
        hints = getattr(scheduler_output, "_ds_preload", None)
        if hints:
            self._preload_adapters(hints)
        return super().execute_model(scheduler_output, *args, **kwargs)

    def _preload_adapters(self, lora_requests) -> None:
        """[cold start] Read these adapters from disk on a background thread
        while this step runs: the next step, which uses them, finds them in
        host memory. vLLM's own ``_load_adapter`` (wrapped once, on the
        object) takes a finished read, waits for one in flight, and reads
        itself whatever was not preloaded — so the result is the same with or
        without the hint, only the step that pays differs."""
        import threading

        mgr = getattr(self.model_runner, "lora_manager", None)
        if mgr is None:
            return
        jobs = getattr(mgr, "_ds_preloads", None)
        if jobs is None:
            jobs = mgr._ds_preloads = {}
            read = mgr._load_adapter

            def load_adapter(lora_request):
                job = jobs.pop(lora_request.lora_int_id, None)
                if job is not None:
                    thread, box = job
                    thread.join()
                    if box.get("lora") is not None \
                            and box["path"] == lora_request.lora_path:
                        return box["lora"]
                return read(lora_request)

            mgr._ds_read_adapter = read
            mgr._load_adapter = load_adapter
        try:
            known = mgr.list_adapters()
        except Exception:  # noqa: BLE001
            known = ()
        for req in lora_requests:
            if req.lora_int_id in jobs or req.lora_int_id in known:
                continue
            box = {"path": req.lora_path}

            def run(req=req, box=box):
                try:
                    box["lora"] = mgr._ds_read_adapter(req)
                except Exception:  # noqa: BLE001 — the step reads it itself
                    box["lora"] = None

            thread = threading.Thread(target=run, daemon=True,
                                      name="deltaserve-adapter-preload")
            jobs[req.lora_int_id] = (thread, box)
            thread.start()

    def deltaserve_route_encode(self, batch: list):
        """The encode step. ``batch``: e5 token-id lists (prefix included,
        truncated by the API server). Returns [n, dim] float32 from the rank
        that holds the encoder, None from the others, ``{"error": ...}`` on a
        failure — never raises (see ``deltaserve_switch_ft_adapter``)."""
        try:
            runner = self.model_runner
            coord = getattr(runner, "_ft_coordinator", None)
            if coord is not None:
                # The encode is the first part of this request's TTFT: the
                # backward child leaves the GPU for it as for a prefill.
                cfg = self.vllm_config.finetune_config
                if coord.pending_backward or cfg.pause_prefill_always:
                    coord.gpu_pause_backward()
                    runner._ds_route_hold = True
            if self._route_encoder is None or not batch:
                return None
            return self._route_forward(batch)
        except Exception as e:  # noqa: BLE001 — reported to the engine
            return {"error": f"{type(e).__name__}: {e}"}


class RouteWorker(RouteEncodeMixin, Worker):
    """Adapter routing without finetuning: the stock worker + the encoder."""
