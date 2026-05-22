import gc
import torch
import torch.nn as nn
from dserve.server.router.mixed_req_queue import rprint
from pprint import pprint

class LoraLayerWeight:
    def __init__(self, layer_num, tp_rank, world_size, lora_config, network_config,
                 data_type=torch.float16, no_lora_swap=False, prefetch_stream=None, is_finetuning=False):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.lora_config = lora_config
        self.network_config = network_config
        self.prefetch_stream = prefetch_stream
        self.no_lora_swap = no_lora_swap
        self.is_finetuning = is_finetuning
        # ---- 1) normalize config (keeps llama1 identical) ----
        if self.network_config.get("num_key_value_heads", None) is None:
            self.network_config["num_key_value_heads"] = self.network_config["num_attention_heads"]

        H = self.network_config["hidden_size"]
        n_q = self.network_config["num_attention_heads"]
        n_kv = self.network_config["num_key_value_heads"]

        assert H % n_q == 0
        head_dim = H // n_q
        kv_out = n_kv * head_dim

        assert n_q % n_kv == 0
        assert H % self.world_size_ == 0
        assert kv_out % self.world_size_ == 0

        self.hidden_size_ = H
        self.num_q_heads_ = n_q
        self.num_kv_heads_ = n_kv
        self.head_dim_ = head_dim
        self.kv_out_dim_ = kv_out

        # "q/o shard width" (what your existing infra assumes for page width)
        self.split_qo_ = H // self.world_size_
        # true kv shard width for GQA
        self.split_kv_ = kv_out // self.world_size_

        self.is_gqa_ = (n_kv != n_q)

        # ---- 2) TP slice ranges ----
        self.tp_q_out_ = (self.split_qo_ * self.tp_rank_, self.split_qo_ * (self.tp_rank_ + 1))
        self.tp_kv_out_ = (self.split_kv_ * self.tp_rank_, self.split_kv_ * (self.tp_rank_ + 1))
        self.tp_o_in_ = (self.split_qo_ * self.tp_rank_, self.split_qo_ * (self.tp_rank_ + 1))

        # ---- 3) init all expected params to None (preserve infra expectations) ----
        self.q_lora_A = self.q_lora_B = None
        self.k_lora_A = self.k_lora_B = None
        self.v_lora_A = self.v_lora_B = None
        self.o_lora_A = self.o_lora_B = None

        # swap/cpu homes (optional)
        self.q_lora_A_home = self.q_lora_B_home = None
        self.k_lora_A_home = self.k_lora_B_home = None
        self.v_lora_A_home = self.v_lora_B_home = None
        self.o_lora_A_home = self.o_lora_B_home = None

        self.w_combined_home = None
        self.w_combined = None
        self.w_combined_home_fp32 = None

    def _dtype_cast(self, t: torch.Tensor) -> torch.Tensor:
        # self.data_type_ is torch dtype in your codebase
        if t.dtype != self.data_type_:
            t = t.to(self.data_type_)
        return t
    def _pad_outdim_A(self, A_out_r: torch.Tensor, target_out: int) -> torch.Tensor:
        """
        A is stored as [out_dim_shard, r] (after transpose in your code).
        Pad rows at the end to reach target_out.
        """
        if A_out_r is None:
            return None
        out_dim, r = A_out_r.shape
        if out_dim == target_out:
            return A_out_r
        if out_dim > target_out:
            # should never happen if target_out is split_qo_
            return A_out_r[:target_out, :]
        pad = target_out - out_dim
        z = torch.zeros((pad, r), dtype=A_out_r.dtype, device=A_out_r.device)
        return torch.cat([A_out_r, z], dim=0)

    def _pad_outdim_B(self, B_r_out: torch.Tensor, target_out: int) -> torch.Tensor:
        """
        B is stored as [r, out_dim_shard] (after transpose in your code).
        Pad cols at the end to reach target_out.
        """
        if B_r_out is None:
            return None
        r, out_dim = B_r_out.shape
        if out_dim == target_out:
            return B_r_out
        if out_dim > target_out:
            return B_r_out[:, :target_out]
        pad = target_out - out_dim
        z = torch.zeros((r, pad), dtype=B_r_out.dtype, device=B_r_out.device)
        return torch.cat([B_r_out, z], dim=1)

    def _build_w_combined_home(self):
        """
        Always build w_combined_home with llama1-compatible shape:
            (2, 4 * rank, num_head(=n_q), head_dim)
        For GQA, K/V have zeros in the extra-head region because we padded
        their out-dim shard to split_qo_ (n_q*head_dim per TP).
        """
        rank = int(self.lora_config["r"])
        num_head = int(self.num_q_heads_)
        head_dim = int(self.head_dim_)

        # safety: require CPU homes exist
        assert self.q_lora_A_home is not None and self.q_lora_B_home is not None
        assert self.k_lora_A_home is not None and self.k_lora_B_home is not None
        assert self.v_lora_A_home is not None and self.v_lora_B_home is not None
        assert self.o_lora_A_home is not None and self.o_lora_B_home is not None

        # Each .T is [r, out_dim_shard] -> reshape(rank, num_head, head_dim)
        self.w_combined_home = torch.concat(
            [
                self.q_lora_A_home.T.reshape(rank, num_head, head_dim),
                self.k_lora_A_home.T.reshape(rank, num_head, head_dim),
                self.v_lora_A_home.T.reshape(rank, num_head, head_dim),
                self.o_lora_A_home.T.reshape(rank, num_head, head_dim),
                self.q_lora_B_home.T.reshape(rank, num_head, head_dim),
                self.k_lora_B_home.T.reshape(rank, num_head, head_dim),
                self.v_lora_B_home.T.reshape(rank, num_head, head_dim),
                self.o_lora_B_home.T.reshape(rank, num_head, head_dim),
            ]
        ).pin_memory()

        self.w_combined_home = self.w_combined_home.reshape(2, 4 * rank, num_head, head_dim)
        self.w_combined = None
        self.w_combined_home_fp32 = None
        if self.is_finetuning:
            self.w_combined_home_fp32 = (
                self.w_combined_home.detach().clone().float().to("cuda", non_blocking=True)
            )
        

    def load_to_torch(self, path):
        numpy_type = {"fp32": np.float32, "fp16": np.float16}[self.data_type_]
        torch_type = {"fp32": torch.float32, "fp16": torch.float16}[self.data_type_]
        return torch.from_numpy(np.fromfile(path, dtype=numpy_type)).to(torch_type)
    
    def items(self):
        if self.q_lora_A is None:
            return None
        return {"q_lora_A": self.q_lora_A, "q_lora_B": self.q_lora_B,
                "k_lora_A": self.k_lora_A, "k_lora_B": self.k_lora_B,
                "v_lora_A": self.v_lora_A, "v_lora_B": self.v_lora_B,
                "o_lora_A": self.o_lora_A, "o_lora_B": self.o_lora_B}

    def load_dummy_weights(self, swap):
        """
        Dummy weights that match your existing w_combined/page-buffer assumption:
          - Q/O always have out_dim_shard = split_qo_
          - K/V true out_dim_shard = split_kv_, but we PAD to split_qo_ for llama3
        """
        rank = int(self.lora_config["r"])

        def rand_gpu(shape):
            return (torch.rand(shape, dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3

        # logical shard widths
        qo = self.split_qo_
        kv = self.split_kv_

        if not swap or self.no_lora_swap:
            # Q/O: [out, r] and [r, out] after transpose behavior
            self.q_lora_A = rand_gpu((rank, qo)).transpose(0, 1).contiguous()
            self.q_lora_B = rand_gpu((qo, rank)).transpose(0, 1).contiguous()

            self.o_lora_A = rand_gpu((rank, qo)).transpose(0, 1).contiguous()
            self.o_lora_B = rand_gpu((qo, rank)).transpose(0, 1).contiguous()

            # K/V: generate true [kv, r] / [r, kv], then pad to [qo, r] / [r, qo] for llama3
            kA = rand_gpu((rank, kv)).transpose(0, 1).contiguous()
            kB = rand_gpu((kv, rank)).transpose(0, 1).contiguous()
            vA = rand_gpu((rank, kv)).transpose(0, 1).contiguous()
            vB = rand_gpu((kv, rank)).transpose(0, 1).contiguous()

            self.k_lora_A = self._pad_outdim_A(kA, qo)
            self.k_lora_B = self._pad_outdim_B(kB, qo)
            self.v_lora_A = self._pad_outdim_A(vA, qo)
            self.v_lora_B = self._pad_outdim_B(vB, qo)
            return

        # ---- swap path: store CPU pinned homes and build w_combined_home ----
        # Generate CPU pinned buffers
        self.q_lora_A_home = rand_gpu((rank, qo)).transpose(0, 1).contiguous().to("cpu").pin_memory()
        self.q_lora_B_home = rand_gpu((qo, rank)).transpose(0, 1).contiguous().to("cpu").pin_memory()

        self.o_lora_A_home = rand_gpu((rank, qo)).transpose(0, 1).contiguous().to("cpu").pin_memory()
        self.o_lora_B_home = rand_gpu((qo, rank)).transpose(0, 1).contiguous().to("cpu").pin_memory()

        kA = rand_gpu((rank, kv)).transpose(0, 1).contiguous().to("cpu").pin_memory()
        kB = rand_gpu((kv, rank)).transpose(0, 1).contiguous().to("cpu").pin_memory()
        vA = rand_gpu((rank, kv)).transpose(0, 1).contiguous().to("cpu").pin_memory()
        vB = rand_gpu((kv, rank)).transpose(0, 1).contiguous().to("cpu").pin_memory()

        # pad to qo width (so reshape(rank, n_q, head_dim) works)
        self.k_lora_A_home = self._pad_outdim_A(kA, qo).pin_memory()
        self.k_lora_B_home = self._pad_outdim_B(kB, qo).pin_memory()
        self.v_lora_A_home = self._pad_outdim_A(vA, qo).pin_memory()
        self.v_lora_B_home = self._pad_outdim_B(vB, qo).pin_memory()

        # Clear GPU refs
        self.q_lora_A = self.q_lora_B = None
        self.k_lora_A = self.k_lora_B = None
        self.v_lora_A = self.v_lora_B = None
        self.o_lora_A = self.o_lora_B = None

        # Always build combined (llama1-compatible shape); KV extra region is zeros for llama3.
        self._build_w_combined_home()
        return

    @torch.no_grad()
    def load_hf_weights(self, weights, swap=False, dummy=False):
        if dummy:
            rprint("is dummy")
            self.load_dummy_weights(swap)
            return

        if swap and not self.no_lora_swap:
            self.load_hf_weights_cpu(weights)
            return

        H = self.hidden_size_
        split_qo = self.split_qo_
        split_kv = self.split_kv_

        # TP slices: Q/O slice over H, K/V slice over kv_out
        tp_qo = (split_qo * self.tp_rank_, split_qo * (self.tp_rank_ + 1))
        tp_kv = (split_kv * self.tp_rank_, split_kv * (self.tp_rank_ + 1))

        prefix = list(weights.keys())[0]
        prefix = prefix[: prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"

        # -------- Q --------
        if f"{prefix}.q_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.q_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            self.q_lora_A = self._dtype_cast(A.transpose(0, 1).contiguous()).cuda()

        if f"{prefix}.q_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.q_proj.lora_B.weight"][tp_qo[0] : tp_qo[1], :]
            self.q_lora_B = self._dtype_cast(B.transpose(0, 1).contiguous()).cuda()

        # -------- K (pad to split_qo for llama3) --------
        if f"{prefix}.k_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.k_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            A = self._dtype_cast(A.transpose(0, 1).contiguous()).cuda()
            self.k_lora_A = self._pad_outdim_A(A, split_qo)

        if f"{prefix}.k_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.k_proj.lora_B.weight"][tp_kv[0] : tp_kv[1], :]
            B = self._dtype_cast(B.transpose(0, 1).contiguous()).cuda()
            self.k_lora_B = self._pad_outdim_B(B, split_qo)

        # -------- V (pad to split_qo for llama3) --------
        if f"{prefix}.v_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.v_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            A = self._dtype_cast(A.transpose(0, 1).contiguous()).cuda()
            self.v_lora_A = self._pad_outdim_A(A, split_qo)

        if f"{prefix}.v_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.v_proj.lora_B.weight"][tp_kv[0] : tp_kv[1], :]
            B = self._dtype_cast(B.transpose(0, 1).contiguous()).cuda()
            self.v_lora_B = self._pad_outdim_B(B, split_qo)

        # -------- O --------
        if f"{prefix}.o_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.o_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            self.o_lora_A = self._dtype_cast(A.transpose(0, 1).contiguous()).cuda()

        if f"{prefix}.o_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.o_proj.lora_B.weight"][tp_qo[0] : tp_qo[1], :]
            self.o_lora_B = self._dtype_cast(B.transpose(0, 1).contiguous()).cuda()

        return

    @torch.no_grad()
    def load_hf_weights_cpu(self, weights):
        """
        CPU+swap path: always build w_combined_home, with KV padded to split_qo.
        Keeps llama1 behavior identical (for llama1 split_kv == split_qo so padding is no-op).
        """
        split_qo = self.split_qo_
        split_kv = self.split_kv_

        tp_qo = (split_qo * self.tp_rank_, split_qo * (self.tp_rank_ + 1))
        tp_kv = (split_kv * self.tp_rank_, split_kv * (self.tp_rank_ + 1))

        prefix = list(weights.keys())[0]
        prefix = prefix[: prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"

        # -------- Q --------
        if f"{prefix}.q_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.q_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            self.q_lora_A_home = self._dtype_cast(A.transpose(0, 1).contiguous()).pin_memory()
            self.q_lora_A = None

        if f"{prefix}.q_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.q_proj.lora_B.weight"][tp_qo[0] : tp_qo[1], :]
            self.q_lora_B_home = self._dtype_cast(B.transpose(0, 1).contiguous()).pin_memory()
            self.q_lora_B = None

        # -------- K (pad to split_qo) --------
        if f"{prefix}.k_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.k_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            A = self._dtype_cast(A.transpose(0, 1).contiguous())
            self.k_lora_A_home = self._pad_outdim_A(A, split_qo).pin_memory()
            self.k_lora_A = None

        if f"{prefix}.k_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.k_proj.lora_B.weight"][tp_kv[0] : tp_kv[1], :]
            B = self._dtype_cast(B.transpose(0, 1).contiguous())
            self.k_lora_B_home = self._pad_outdim_B(B, split_qo).pin_memory()
            self.k_lora_B = None

        # -------- V (pad to split_qo) --------
        if f"{prefix}.v_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.v_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            A = self._dtype_cast(A.transpose(0, 1).contiguous())
            self.v_lora_A_home = self._pad_outdim_A(A, split_qo).pin_memory()
            self.v_lora_A = None

        if f"{prefix}.v_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.v_proj.lora_B.weight"][tp_kv[0] : tp_kv[1], :]
            B = self._dtype_cast(B.transpose(0, 1).contiguous())
            self.v_lora_B_home = self._pad_outdim_B(B, split_qo).pin_memory()
            self.v_lora_B = None

        # -------- O --------
        if f"{prefix}.o_proj.lora_A.weight" in weights:
            A = weights[f"{prefix}.o_proj.lora_A.weight"][:, tp_qo[0] : tp_qo[1]]
            self.o_lora_A_home = self._dtype_cast(A.transpose(0, 1).contiguous()).pin_memory()
            self.o_lora_A = None

        if f"{prefix}.o_proj.lora_B.weight" in weights:
            B = weights[f"{prefix}.o_proj.lora_B.weight"][tp_qo[0] : tp_qo[1], :]
            self.o_lora_B_home = self._dtype_cast(B.transpose(0, 1).contiguous()).pin_memory()
            self.o_lora_B = None

        # Always build combined (KV zeros for llama3).
        self._build_w_combined_home()
        return

    def load_to_gpu(self, prefetch=False, bmm=False, both=False):
        """
          - If not bmm: move w_combined_home -> w_combined
          - If bmm: move per-matrix homes -> GPU
        """
        if not bmm:
            if self.w_combined is None:
                self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)
                if not torch.isfinite(self.w_combined).all():
                    print(f"⚠️ [corrupt param] {self.w_combined} contains inf or NaN")
            else:
                self.w_combined = self.w_combined.to("cuda", non_blocking=True)
        else:
            if self.q_lora_A is None:
                self.q_lora_A = self.q_lora_A_home.to("cuda", non_blocking=True)
                self.q_lora_B = self.q_lora_B_home.to("cuda", non_blocking=True)
                self.k_lora_A = self.k_lora_A_home.to("cuda", non_blocking=True)
                self.k_lora_B = self.k_lora_B_home.to("cuda", non_blocking=True)
                self.v_lora_A = self.v_lora_A_home.to("cuda", non_blocking=True)
                self.v_lora_B = self.v_lora_B_home.to("cuda", non_blocking=True)
                self.o_lora_A = self.o_lora_A_home.to("cuda", non_blocking=True)
                self.o_lora_B = self.o_lora_B_home.to("cuda", non_blocking=True)
            if both and self.w_combined_home is not None:
                self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)

    def offload_from_gpu(self):
        if self.no_lora_swap:
            return
        self.w_combined = None
        self.q_lora_A = None
        self.q_lora_B = None
        self.k_lora_A = None
        self.k_lora_B = None
        self.v_lora_A = None
        self.v_lora_B = None
        self.o_lora_A = None
        self.o_lora_B = None