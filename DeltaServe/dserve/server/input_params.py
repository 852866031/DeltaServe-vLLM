"""
Thin views over a ServerConfig. Public attributes are kept stable so the ~50
downstream readers (`self.input_params.foo`, `self.finetuning_params.bar`)
do not need to change. Future PRs can incrementally rename call sites to read
`cfg.section.field` directly and then delete these wrappers.
"""
from dserve.server.config import ServerConfig


class FinetuneParams:
    def __init__(self, cfg: ServerConfig):
        ft = cfg.finetune
        # Downstream uses `finetuning_lora_path is not None` as the kill switch
        # for the entire finetune machinery — preserve that behavior by gating
        # the path-bearing fields on `ft.enabled`.
        active = ft.enabled
        self.finetuning_type = ft.type if active else None
        self.finetuning_data_path = ft.data_path if active else None
        self.finetuning_lora_path = ft.lora_path if active else None
        self.learning_rate = ft.learning_rate
        self.weight_decay = ft.weight_decay
        self.gamma = ft.gamma
        self.num_epochs = ft.num_epochs
        self.max_saved_finetuning_tokens = ft.max_saved_finetuning_tokens
        self.max_finetuning_tokens_in_batch = ft.max_finetuning_tokens_in_batch
        self.optimizer_threading = ft.optimizer_threading
        self.start_on_launch = ft.start_on_launch
        self.finetuning_prepare_size = ft.prepare_size
        self.ft_log_path = ft.log_path
        # Legacy non-finetune fields stashed here historically; kept until
        # downstream callers are migrated.
        self.model_weightdir = cfg.model.dir
        self.tokenizor_mode = cfg.model.tokenizer_mode
        self.trust_remote_code = cfg.model.trust_remote_code
        # Eval steps stays at the historical default; not exposed in YAML.
        self.eval_steps = 100


class SLOParams:
    def __init__(self, cfg: ServerConfig):
        slo = cfg.slo
        self.ttft_slo = slo.ttft_slo
        self.avg_tbt_slo = slo.avg_tbt_slo
        self.max_tbt_slo = slo.max_tbt_slo


class InputParams:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg

        # serving
        self.max_req_total_len = cfg.serving.max_req_total_len
        self.max_total_token_num = cfg.serving.max_total_token_num
        self.batch_max_tokens = cfg.serving.batch_max_tokens
        self.running_max_req_size = cfg.serving.running_max_req_size

        # lora
        self.pool_size_lora = cfg.lora.pool_size_lora
        self.swap = cfg.lora.swap
        self.prefetch = cfg.lora.prefetch
        self.prefetch_size = cfg.lora.prefetch_size
        self.batch_num_adapters = cfg.lora.batch_num_adapters
        self.fair_weights = list(cfg.lora.fair_weights)

        # scheduler
        self.scheduler = cfg.scheduler.name
        self.enable_abort = cfg.scheduler.enable_abort

        # cuda graph (legacy attribute names preserved for downstream callers)
        self.enable_cuda_graph = cfg.cuda_graph.enable_decode_cuda_graph
        self.enable_prefill_cuda_graph = cfg.cuda_graph.enable_prefill_cuda_graph
        self.enable_bwd_cuda_graph = cfg.cuda_graph.enable_bwd_cuda_graph

        # debug
        self.dummy = cfg.debug.dummy
        self.no_lora = cfg.debug.no_lora
        self.no_lora_compute = cfg.debug.no_lora_compute
        self.no_lora_swap = cfg.debug.no_lora_swap
        self.no_kernel = cfg.debug.no_kernel
        self.no_mem_pool = cfg.debug.no_mem_pool
        self.bmm = cfg.debug.bmm
        self.profile = cfg.debug.profile

        self.finetuning_params = FinetuneParams(cfg)
        self.slo_params = SLOParams(cfg)
