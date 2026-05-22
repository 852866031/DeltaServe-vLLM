from enum import Enum
import hashlib
import math
import time
from multiprocessing import Pipe
from dserve.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from dserve.models.llama.triton_kernel.rmsnorm import rmsnorm_backward, rmsnorm_forward

def bwd_print(*args, sep=' ', end='\n'):
    color = "\033[35m"
    RESET = "\033[0m"
    text = sep.join(str(arg) for arg in args)
    print(f"{color}[BWD]: {text}{RESET}", end=end)

class BaseModelWeights():
    def __init__(self):
        self.pre_post_weight = None
        self.trans_layers_weight = []
        self._cos_cached = None
        self._sin_cached = None

class AdapterWeights():
    def __init__(self):
        self.scaling = None
        self.lora_weights = []

class Activations():
    def __init__(self):
        self.logit_list = []
        self.concat_input_ids = None
        self.transformer_out_activations = []
        self.attention_out_activations = []
        self.input_layer_output = None
        # CPU-side per-request sequence lengths, populated by get_logits_and_targets.
        # Used by _backpop_attention to drive Python-side loops without D→H syncs.
        self.seq_lens_py: list = []
        self.b_start_py: list = []

class SharedActivations():
    def __init__(self):
        self.logit_tensor = None
        self.logit_list = []
        self.concat_input_ids = None
        self.transformer_out_activations = []
        self.attention_out_activations = []
        self.input_layer_output = None


def tensor_hash(t: torch.Tensor, algo="sha256") -> str:
    h = hashlib.new(algo)
    h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class LlamaSFTBackwardService():
    def __init__(self, network_config, recv_pipe, send_pipe, lr=1e-4, weight_decay=0.01, gamma=0.95, use_rank_id=0, bwd_log_index=0, enable_bwd_graph=False, max_saved_finetuning_tokens=512,
                 use_graphed_bwd_attention=True, attn_bn_max=8, attn_l_max=64):
        # When True, _lora_context_backward dispatches to _backpop_attention_padded
        # (a graph-capturable, fixed-shape variant) if the subclass provides one.
        # Otherwise the original variable-shape _backpop_attention is used. Only
        # llama3 (Llama3SFTBackwardService) defines _backpop_attention_padded;
        # base llama has no such method, so flipping this flag is a no-op for it.
        self.USE_GRAPHED_ATTENTION = use_graphed_bwd_attention
        # Fixed-shape padding budget for _backpop_attention_padded. Every real
        # fine-tuning batch must satisfy: Bn <= ATTN_BN_MAX and
        # max(seq_lens) <= ATTN_L_MAX. Used by the llama3 subclass; harmless
        # to set here.
        self.ATTN_BN_MAX = attn_bn_max
        self.ATTN_L_MAX = attn_l_max
        # Runtime references / constants
        self.eps_ = network_config["rms_norm_eps"]
        self.embed_dim_ = network_config["hidden_size"]
        self.num_layers = network_config["n_layer"]
        self.last_loss = None #debugging
        self.model_weights = BaseModelWeights()
        self.adapter_weights = AdapterWeights()
        self.activations = None
        self.shared_activations = SharedActivations()
        self.recv_pipe = recv_pipe
        self.send_pipe = send_pipe
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.current_epoch = 0
        self.bwd_stream = None
        self.bwd_pause_event = None
        self.working = False
        self.total_processed_tokens = 0
        self.ft_log = []
        self.log_batch_counter = {}   # epoch → counter
        self.use_rank_id = use_rank_id
        self.max_saved_finetuning_tokens = max_saved_finetuning_tokens
        self.enable_bwd_graph = enable_bwd_graph
        self.graph_runner = None
        if enable_bwd_graph:
            from dserve.models.llama.SFT_service_graph import GraphedBackwardRunner
            self.graph_runner = GraphedBackwardRunner(self)

    def start_service(self):
        torch.cuda.set_device(self.use_rank_id)
        bwd_print("Started. Using rank id:", self.use_rank_id)
        msg = self.recv_pipe.recv()
        self.receive_adapter(msg)
        msg = self.recv_pipe.recv()
        self.receive_activation_addresses(msg)
        self.finetuning_optimizer = torch.optim.AdamW(
                self.adapter_weights.lora_weights, 
                lr=self.lr, 
                betas=(0.9,0.999), 
                weight_decay=self.weight_decay)
        self.finetuning_scheduler = torch.optim.lr_scheduler.StepLR(
            self.finetuning_optimizer,
            step_size=1,      # every epoch
            gamma=self.gamma        # multiply by 0.5
        )
        self.bwd_stream = torch.cuda.Stream()
        if self.graph_runner is not None:
            self.graph_runner.prepare()
        self.send_pipe.send("READY")
        while True:
            if self.recv_pipe.poll():
                msg = self.recv_pipe.recv()
                if msg == "EXIT":
                    bwd_print("Shutting down.")
                    break
                elif isinstance(msg, dict):
                    try:
                        self.receive_requests_info(msg)
                        current_epoch = msg["current_epoch"]
                        #bwd_print(f"Received activations. Starting backward...")
                        self.working = True
                        self._maybe_pause()
                        with torch.cuda.stream(self.bwd_stream):
                            start = time.time()
                            ok, loss, ntok = self._run_bwd()
                            #torch.cuda.synchronize()
                            self._maybe_pause()
                            self.finetuning_optimizer.step()
                            self.finetuning_optimizer.zero_grad(set_to_none=True)
                            if current_epoch > self.current_epoch:
                                self.finetuning_scheduler.step()
                                self.current_epoch = current_epoch
                            self.send_pipe.send((ok, loss, ntok))
                        self.working = False
                        self.total_processed_tokens += ntok
                        bwd_print(f"Backward completed. Total tokens processed: {self.total_processed_tokens}, loss: {loss:.6f}")
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        self.send_pipe.send({"error": str(e), "traceback": tb})
                        bwd_print(f"Backward failed with error: {e}\n{tb}")
            else:
                time.sleep(0.1)  # yield to CPU, avoid busy wait
        
    def _maybe_pause(self, drain_stream=True):
        if getattr(self, "bwd_pause_event", None) is None or self.bwd_pause_event.is_set():
            return
        if self.working: bwd_print("Paused")
        while not self.bwd_pause_event.is_set():
            time.sleep(0.1)
        if self.working: bwd_print("Resumed")

    def receive_model_dict(self, model_dict):
        self.model_weights.pre_post_weight = model_dict["pre_post_weight"]
        self.model_weights.trans_layers_weight = model_dict["trans_layers_weight"][:]
        self.model_weights._cos_cached = model_dict["_cos_cached"]
        self.model_weights._sin_cached = model_dict["_sin_cached"]
        return

    def receive_adapter(self, adapter_dict):
        if self.adapter_weights is not None:
            del self.adapter_weights
        self.adapter_weights = AdapterWeights()
        self.adapter_weights.scaling = adapter_dict["scaling"]
        self.adapter_weights.lora_weights = adapter_dict["lora_weights"][:]
        # for layer in self.adapter_weights.lora_weights:
        #     layer.requires_grad = True
        return


    def receive_activation_addresses(self, activations_dict):
        self.shared_activations = SharedActivations()
        self.shared_activations.logit_tensor = activations_dict["logit_tensor"]
        self.shared_activations.concat_input_ids = activations_dict["concat_input_ids"]
        self.shared_activations.transformer_out_activations = activations_dict["transformer_out_activations"]
        self.shared_activations.attention_out_activations = activations_dict["attention_out_activations"]
        self.shared_activations.input_layer_output = activations_dict["input_layer_output"]
    
    def receive_requests_info(self, dict):
        if self.activations is not None:
            del self.activations
        self.activations = Activations()
        request_token_info = dict["request_token_info"]
        total_token_num = sum(request_token_info)
        
        self.activations.logit_list = []
        logit_tensor = self.shared_activations.logit_tensor[:total_token_num].detach().clone()
        for n in request_token_info:
            logit = logit_tensor[:n, :]
            self.activations.logit_list.append(logit)
            logit_tensor = logit_tensor[n:, :]

        self.activations.concat_input_ids = self.shared_activations.concat_input_ids[:total_token_num+len(request_token_info)].clone()
        self.activations.transformer_out_activations = []
        self.activations.attention_out_activations = []
        for i in range(self.num_layers):
            self.activations.transformer_out_activations.append(self.shared_activations.transformer_out_activations[i][:total_token_num])
            self.activations.attention_out_activations.append(self.shared_activations.attention_out_activations[i][:total_token_num])
        self.activations.input_layer_output = self.shared_activations.input_layer_output[:total_token_num]
        return


    def _run_bwd(self):
        if self.graph_runner is not None:
            try:
                return self.graph_runner.run()
            except Exception as e:
                bwd_print(f"Graph runner failed, disabling and falling back to eager: {e}")
                import traceback
                traceback.print_exc()
                self.graph_runner = None
        return self._context_backward()

    def _context_backward(self):
        logits_and_targets, total_tokens_to_process = self.get_logits_and_targets()
        loss = self.compute_total_loss(logits_and_targets)
        bwd_print(f"Backward Total Tokens: {total_tokens_to_process}, Loss: {loss:.12f}")
        logit_grad = self._logit_backward(logits_and_targets)
        self._maybe_pause()
        grad_transformer_out = self._post_layer_backward(logit_grad, self.model_weights.pre_post_weight)
        for i in reversed(range(self.num_layers)):
            self._maybe_pause()
            grad_transformer_out = self._lora_context_backward(i, grad_transformer_out)
        return True, loss, total_tokens_to_process

    def get_logits_and_targets(self):
        logits_list = self.activations.logit_list
        seq_lens_py = [int(logits.shape[0]) for logits in logits_list]
        # Stash CPU-side copies so the attention backward can drive its
        # per-request loop without D→H syncs.
        self.activations.seq_lens_py = seq_lens_py
        b_start_py = [0] * len(seq_lens_py)
        for i in range(1, len(seq_lens_py)):
            b_start_py[i] = b_start_py[i - 1] + seq_lens_py[i - 1]
        self.activations.b_start_py = b_start_py
        input_ids = self.activations.concat_input_ids
        total_tokens_to_process = input_ids.shape[0]

        # Compute per-request input length from logits (add 1 to get back original token count)
        token_counts = [logits.shape[0] + 1 for logits in logits_list]  # [T_i - 1] → [T_i]
        input_ids_split = torch.split(input_ids, token_counts)

        # Align logits and targets
        results = []
        for logits, input_ids_i in zip(logits_list, input_ids_split):
            if logits.shape[0] < 1:
                continue  # skip empty logits

            # Cast to float32 for numerical stability
            pred_logits = logits.float()          # [T-1, vocab]
            target_ids = input_ids_i[1:].long()   # [T-1], ensure targets are long for CE loss

            results.append((pred_logits, target_ids))

        return results, total_tokens_to_process

    # For SFT, compute the total loss of the batch
    def compute_total_loss(self, logits_and_targets, ignore_index=-100):
        total_loss = 0.0
        total_tokens = 0
        for logits, targets in logits_and_targets:
            if logits.shape[0] != targets.shape[0]:
                raise ValueError(f"Logits and targets length mismatch: {logits.shape[0]} vs {targets.shape[0]}")

            # Compute CE loss   
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100, reduction='sum')
            total_loss += loss.item()
            valid_mask = targets != ignore_index
            total_tokens += valid_mask.sum().item()
        if total_tokens == 0:
            return torch.tensor(0.0)

        losses = torch.tensor(total_loss / total_tokens)

        if self.last_loss and torch.equal(losses, self.last_loss):
            print("losses equal to last loss")
        self.last_loss = losses.clone()


        return losses
    
    # Backprop the gradient to the logit 
    def _logit_backward(self, logits_and_targets):
        all_logits = []
        all_targets = []

        for logits, targets in logits_and_targets:
            all_logits.append(logits)      # each [T_i - 1, vocab]
            all_targets.append(targets)    # each [T_i - 1]

        logits_cat = torch.cat(all_logits, dim=0)   # [N, vocab]
        targets_cat = torch.cat(all_targets, dim=0) # [N]

        # Compute ∇loss/∇logits manually (cross-entropy gradient)
        probs = torch.softmax(logits_cat, dim=-1)  # [N, vocab]
        probs[torch.arange(len(targets_cat)), targets_cat] -= 1
        probs /= len(targets_cat)  # normalize over total number of tokens
        return probs  # gradient of loss w.r.t logits: [N, vocab]


    # Backprop the gradient to the output layer (original fp32 path).
    # Kept as a reference / fallback. Not wired into _context_backward.
    def _post_layer_backward(self,
                                  logit_grad: torch.Tensor,
                                  layer_weight) -> torch.Tensor:
        lm_W     = layer_weight.lm_head_weight_.float()      # (V, D)
        norm_W   = layer_weight.final_norm_weight_.float()   # (D,)
        x = self.activations.transformer_out_activations[-1]                  # (N, D)
        g_y = logit_grad @ lm_W            # (N, D)
        D = x.shape[-1]
        r = x.norm(p=2, dim=-1, keepdim=True) / math.sqrt(D)   # (N, 1)
        g_x_hat = g_y * norm_W               # (N, D)
        dot     = (g_x_hat * x).sum(dim=-1, keepdim=True)      # (N, 1)
        grad_x = (g_x_hat - x * dot / (D * (r**2 + self.eps_))) / (r + self.eps_)
        return grad_x
    
    # Backprop through LoRA augmented transformer layer
    def _lora_context_backward(self, layer_id, output_grad: torch.Tensor):
        # backprop for a transformer layer, start from ffn
        layer_weight = self.model_weights.trans_layers_weight[layer_id]
        ffn_input = self.activations.attention_out_activations[layer_id]      # shape [N, D]
        grad_ffn_input = self._backprop_ffn(ffn_input, output_grad, layer_weight)
        if layer_id == 0:
            last_layer_input = self.activations.input_layer_output    # shape [N, D]
        else:
            last_layer_input = self.activations.transformer_out_activations[layer_id-1]    # shape [N, D]
        # Backprop through LoRA
        self._maybe_pause()
        # Dispatch: graph-capturable padded variant (if the subclass provides
        # one and USE_GRAPHED_ATTENTION is set) vs. the original variable-shape
        # attention backward.
        if self.USE_GRAPHED_ATTENTION and hasattr(self, "_backpop_attention_padded"):
            grad_attn_input = self._backpop_attention_padded(last_layer_input, grad_ffn_input, layer_weight, layer_id)
        else:
            grad_attn_input = self._backpop_attention(last_layer_input, grad_ffn_input, layer_weight, layer_id)
        return grad_attn_input
    
    
    @torch.no_grad()
    def _backprop_ffn(
        self,
        ffn_input: torch.Tensor,          # x  (dtype can be fp16/bf16/fp32)
        output_grad: torch.Tensor,        # ∂L/∂y  with y = x + FFN(...)
        layer_weight,
    ):
        eps = self.eps_
        # --- Keep RMSNorm weights in fp32 for numerical stability ---
        w_rms = layer_weight.ffn_norm_weight_
        w_gate = layer_weight.gate_proj
        w_up   = layer_weight.up_proj
        w_down = layer_weight.down_proj
        x = ffn_input
        dy = output_grad.to(torch.float16)
        x_norm = rmsnorm_forward(x, w_rms, eps=eps)
        # --- Matmuls and activations in bf16/fp16 autocast region ---
        gate_in = x_norm @ w_gate         # [N, D_gate]
        gate_out = torch.nn.functional.silu(gate_in)
        up_out = x_norm @ w_up            # [N, D_up]

        # Backward part
        grad_ffn_mid = dy @ w_down.T      # [N, D_mid]
        grad_gate_out = grad_ffn_mid * up_out
        grad_up_out = grad_ffn_mid * gate_out
        grad_x_norm_up = grad_up_out @ w_up.T
        sig = torch.sigmoid(gate_in)
        silu_grad = sig * (1 + gate_in * (1 - sig))
        grad_gate_in = grad_gate_out * silu_grad
        grad_x_norm_gate = grad_gate_in @ w_gate.T
        grad_x_norm = grad_x_norm_up + grad_x_norm_gate
        grad_from_norm = rmsnorm_backward(x, grad_x_norm, w_rms, eps=eps)
        grad_ffn_input = grad_from_norm + dy
        return grad_ffn_input
    
    @torch.no_grad()
    def _backpop_attention(
        self,
        last_layer_input: torch.Tensor,
        grad_ffn_input: torch.Tensor,
        layer_weight: LlamaTransformerLayerWeight,
        layer_id: int,
    ):
        device = last_layer_input.device

        # CPU-side per-request metadata, precomputed in get_logits_and_targets.
        # All Python-side loop bounds and slice indices below read from these
        # lists, so no D→H syncs are needed inside this function.
        seq_lens_py = self.activations.seq_lens_py
        b_start_py = self.activations.b_start_py

        # ----- Positions -----
        position_ids = torch.cat([
            torch.arange(0, ln, device=device, dtype=torch.long)
            for ln in seq_lens_py
        ])
        position_cos = self.model_weights._cos_cached.index_select(0, position_ids)   # [S, Hd/2], fp16
        position_sin = self.model_weights._sin_cached.index_select(0, position_ids)

        # ----- Weights (fp16) -----
        w_q = layer_weight.q_weight_           # [D, D]
        w_k = layer_weight.k_weight_
        w_v = layer_weight.v_weight_
        w_o = layer_weight.o_weight_
        w_attn_norm = layer_weight.att_norm_weight_

        # ----- 1) RMSNorm forward -----
        x_prev = last_layer_input  # [S, D], fp16
        x_norm = rmsnorm_forward(x_prev, w_attn_norm, eps=self.eps_)  # [S, D], fp16

        # ----- 2) LoRA unpack -----
        w_combined_leaf = self.adapter_weights.lora_weights[layer_id].to(torch.float16)   # [2, 4r, H, Hd], fp16
        r = w_combined_leaf.shape[1] // 4
        H, Hd = w_combined_leaf.shape[2], w_combined_leaf.shape[3]
        D = H * Hd

        # Unpack packed LoRA weights (all fp16)
        qA = w_combined_leaf[0, 0:r].reshape(r, -1).T      # [D, r]
        qB = w_combined_leaf[1, 0:r].reshape(-1, r).T      # [r, D]
        kA = w_combined_leaf[0, r:2*r].reshape(r, -1).T
        kB = w_combined_leaf[1, r:2*r].reshape(-1, r).T
        vA = w_combined_leaf[0, 2*r:3*r].reshape(r, -1).T
        vB = w_combined_leaf[1, 2*r:3*r].reshape(-1, r).T
        oA = w_combined_leaf[0, 3*r:4*r].reshape(r, -1).T  # [D, r]
        oB = w_combined_leaf[1, 3*r:4*r].reshape(-1, r).T  # [r, D]

        scale_lora = self.adapter_weights.scaling

        def proj_lora(X, A, B):
            # X: [S, D], A: [D, r], B: [r, D]
            return (X @ A @ B) * scale_lora

        # Small rotary fwd/bwd in pure PyTorch (fp16)
        def rotary_emb_fwd_pt(q, cos, sin):
            # q: [S, H, Hd], cos/sin: [S, Hd/2]
            S_loc, Hh, Dd = q.shape
            Dh = Dd // 2
            q_flat = q.reshape(S_loc * Hh, Dd)
            q_even = q_flat[:, :Dh]
            q_odd  = q_flat[:, Dh:]
            cos_ex = cos[:, None, :].expand(S_loc, Hh, Dh).reshape(S_loc * Hh, Dh)
            sin_ex = sin[:, None, :].expand(S_loc, Hh, Dh).reshape(S_loc * Hh, Dh)
            q_even_orig = q_even.clone()
            q_even.mul_(cos_ex).addcmul_(q_odd, sin_ex, value=-1.0)
            q_odd.mul_(cos_ex).addcmul_(q_even_orig, sin_ex)

        def rotary_emb_bwd_pt(grad_rot, cos, sin):
            # grad_rot: [S, H, Hd]
            S_loc, Hh, Dd = grad_rot.shape
            Dh = Dd // 2
            g_flat = grad_rot.reshape(S_loc * Hh, Dd)
            g_even = g_flat[:, :Dh]
            g_odd  = g_flat[:, Dh:]
            cos_ex = cos[:, None, :].expand(S_loc, Hh, Dh).reshape(S_loc * Hh, Dh)
            sin_ex = sin[:, None, :].expand(S_loc, Hh, Dh).reshape(S_loc * Hh, Dh)
            # inverse of forward linear transform
            dx_even = g_even * cos_ex + g_odd * sin_ex
            dx_odd  = -g_even * sin_ex + g_odd * cos_ex
            out = torch.empty_like(g_flat)
            out[:, :Dh] = dx_even
            out[:, Dh:] = dx_odd
            return out.reshape(S_loc, Hh, Dd)

        # ----- 3) Q,K,V forward (+LoRA) -----
        X = x_norm.view(-1, D)        # [S, D], fp16
        S_total = X.shape[0]

        q_base = X @ w_q              # [S, D]
        k_base = X @ w_k
        v_base = X @ w_v

        q_ = q_base + proj_lora(X, qA, qB)  # [S, D]
        k_ = k_base + proj_lora(X, kA, kB)
        v_ = v_base + proj_lora(X, vA, vB)

        qh = q_.view(S_total, H, Hd)  # [S, H, Hd]
        kh = k_.view(S_total, H, Hd)
        rotary_emb_fwd_pt(qh, position_cos, position_sin)
        rotary_emb_fwd_pt(kh, position_cos, position_sin)
        vh = v_.view(S_total, H, Hd)

        # ----- 4) Masked causal attention forward -----
        ctx = torch.empty_like(qh)    # [S, H, Hd]
        Bn = len(seq_lens_py)
        scale = 1.0 / (Hd ** 0.5)

        for i in range(Bn):
            st = b_start_py[i]
            ln = seq_lens_py[i]
            q_blk = qh[st:st+ln].transpose(0, 1)   # [H, L, D_h]
            k_blk = kh[st:st+ln].transpose(0, 1)
            v_blk = vh[st:st+ln].transpose(0, 1)

            scores = (q_blk @ k_blk.transpose(-1, -2)) * scale  # [H, L, L]
            mask = torch.triu(torch.ones_like(scores), 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            att = torch.softmax(scores, dim=-1)                 # [H, L, L]

            ctx_blk = (att @ v_blk).transpose(0, 1)             # [L, H, D_h]
            ctx[st:st+ln] = ctx_blk

        ctx_flat = ctx.reshape(S_total, D)  # [S, D]

        # ----- 5) O projection (+LoRA) forward -----
        o_base = ctx_flat @ w_o            # [S, D]
        Zo = ctx_flat @ oA                 # [S, r]
        o_lora = (Zo @ oB) * scale_lora    # [S, D]
        o_total = o_base + o_lora          # [S, D]

        # Residual: output = x_prev + o_total
        grad_o = grad_ffn_input            # [S, D], fp16

        # ---------------- MANUAL BACKWARD ----------------

        # Residual
        grad_x_prev_resid = grad_o.clone()         # dL/d x_prev (resid path)
        grad_o_total = grad_o                      # dL/d o_total

        # --- O base and LoRA O ---
        grad_ctx_from_o = grad_o_total @ w_o.t()   # [S, D]

        grad_Zo = (grad_o_total @ oB.t()) * scale_lora      # [S, r]
        grad_oA = ctx_flat.t() @ grad_Zo                    # [D, r]
        grad_ctx_from_lora_o = grad_Zo @ oA.t()             # [S, D]
        grad_oB = Zo.t() @ (grad_o_total * scale_lora)      # [r, D]

        grad_ctx_flat = grad_ctx_from_o + grad_ctx_from_lora_o   # [S, D]

        # --- Attention backward ---
        grad_qh = torch.zeros_like(qh)   # [S, H, Hd]
        grad_kh = torch.zeros_like(kh)
        grad_vh = torch.zeros_like(vh)

        for i in range(Bn):
            st = b_start_py[i]
            ln = seq_lens_py[i]

            ctx_blk = ctx[st:st+ln]                      # [L, H, Hd]
            g_ctx_blk = grad_ctx_flat[st:st+ln]          # [L, D]
            g_ctx_blk = g_ctx_blk.view(ln, H, Hd)        # [L, H, Hd]

            q_blk = qh[st:st+ln].transpose(0, 1)         # [H, L, Hd]
            k_blk = kh[st:st+ln].transpose(0, 1)
            v_blk = vh[st:st+ln].transpose(0, 1)

            # Recompute att
            scores = (q_blk @ k_blk.transpose(-1, -2)) * scale   # [H, L, L]
            mask = torch.triu(torch.ones_like(scores), 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            att = torch.softmax(scores, dim=-1)                  # [H, L, L]

            g_ctx_T = g_ctx_blk.transpose(0, 1)                  # [H, L, Hd]

            # dV = att^T @ g_ctx
            dV = att.transpose(-1, -2) @ g_ctx_T                 # [H, L, Hd]

            # dAtt = g_ctx @ v^T
            dAtt = g_ctx_T @ v_blk.transpose(-1, -2)             # [H, L, L]

            # softmax backward: dScores = (dAtt - sum(dAtt*att))*att
            s = (dAtt * att).sum(dim=-1, keepdim=True)           # [H, L, 1]
            dScores = (dAtt - s) * att                           # [H, L, L]
            dScores = dScores.masked_fill(mask, 0.0)

            # scores = (q @ k^T) * scale
            dQ = (dScores @ k_blk) * scale                       # [H, L, Hd]
            dK = (dScores.transpose(-1, -2) @ q_blk) * scale     # [H, L, Hd]

            grad_vh[st:st+ln] += dV.transpose(0, 1)              # [L, H, Hd]
            grad_qh[st:st+ln] += dQ.transpose(0, 1)
            grad_kh[st:st+ln] += dK.transpose(0, 1)

        # --- Rotary backward ---
        grad_q_before = rotary_emb_bwd_pt(grad_qh, position_cos, position_sin)
        grad_k_before = rotary_emb_bwd_pt(grad_kh, position_cos, position_sin)

        gq_flat = grad_q_before.reshape(S_total, D)
        gk_flat = grad_k_before.reshape(S_total, D)
        gv_flat = grad_vh.reshape(S_total, D)

        # --- Back through base Q,K,V projections ---
        grad_X_from_q = gq_flat @ w_q.t()
        grad_X_from_k = gk_flat @ w_k.t()
        grad_X_from_v = gv_flat @ w_v.t()

        # --- LoRA Q ---
        Zq = X @ qA                                        # [S, r]
        grad_qB = Zq.t() @ (gq_flat * scale_lora)         # [r, D]
        grad_Zq = (gq_flat * scale_lora) @ qB.t()         # [S, r]
        grad_qA = X.t() @ grad_Zq                         # [D, r]
        grad_X_from_lora_q = grad_Zq @ qA.t()             # [S, D]

        # --- LoRA K ---
        Zk = X @ kA
        grad_kB = Zk.t() @ (gk_flat * scale_lora)         # [r, D]
        grad_Zk = (gk_flat * scale_lora) @ kB.t()         # [S, r]
        grad_kA = X.t() @ grad_Zk                         # [D, r]
        grad_X_from_lora_k = grad_Zk @ kA.t()             # [S, D]

        # --- LoRA V ---
        Zv = X @ vA
        grad_vB = Zv.t() @ (gv_flat * scale_lora)         # [r, D]
        grad_Zv = (gv_flat * scale_lora) @ vB.t()         # [S, r]
        grad_vA = X.t() @ grad_Zv                         # [D, r]
        grad_X_from_lora_v = grad_Zv @ vA.t()             # [S, D]

        # Total grad wrt X (= x_norm)
        grad_X = (grad_X_from_q + grad_X_from_k + grad_X_from_v +
                grad_X_from_lora_q + grad_X_from_lora_k + grad_X_from_lora_v)  # [S, D]

        # --- RMSNorm backward ---
        grad_from_norm = rmsnorm_backward(x_prev, grad_X, w_attn_norm, eps=self.eps_)  # [S, D]

        grad_last_layer_input = grad_from_norm + grad_x_prev_resid   # [S, D]

        # ----------------- pack LoRA grads back to w_combined_leaf.grad -----------
        if w_combined_leaf.grad is None:
            w_combined_leaf.grad = torch.zeros_like(w_combined_leaf)

        # helper: map [D, r] or [r, D] → [r, H, Hd]
        def pack_G(G, transpose_first: bool) -> torch.Tensor:
            if transpose_first:
                G = G.t()            # make it [r, D]
            return G.reshape(r, H, Hd)

        # qA / qB
        w_combined_leaf.grad[0, 0:r] += pack_G(grad_qA, True)   # [r,H,Hd]
        w_combined_leaf.grad[1, 0:r] += pack_G(grad_qB, False)

        # kA / kB
        w_combined_leaf.grad[0, r:2*r] += pack_G(grad_kA, True)
        w_combined_leaf.grad[1, r:2*r] += pack_G(grad_kB, False)

        # vA / vB
        w_combined_leaf.grad[0, 2*r:3*r] += pack_G(grad_vA, True)
        w_combined_leaf.grad[1, 2*r:3*r] += pack_G(grad_vB, False)

        # oA / oB
        w_combined_leaf.grad[0, 3*r:4*r] += pack_G(grad_oA, True)
        w_combined_leaf.grad[1, 3*r:4*r] += pack_G(grad_oB, False)

        # gradient clipping on LoRA tensor (fp16) — branchless so no D→H sync.
        g = w_combined_leaf.grad
        max_norm = 1.0
        gn = g.norm()
        clip_scale = torch.clamp(max_norm / (gn + 1e-6), max=1.0)
        g.mul_(clip_scale)
        self.adapter_weights.lora_weights[layer_id].grad = g.to(torch.float32)
        return grad_last_layer_input