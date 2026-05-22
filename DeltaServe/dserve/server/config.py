"""
Typed serving configuration, loaded from YAML.

This is the future single source of truth for every knob the server takes at
startup. The dataclass hierarchy mirrors the YAML sections 1:1 so a field
`memory.max_finetuning_tokens` in the YAML is reachable as
`cfg.memory.max_finetuning_tokens` in code.

Usage:
    cfg = load_config("path/to/serving_config.yaml")
    apply_overrides(cfg, ["serving.max_total_token_num=30000",
                          "cuda_graph.enable_bwd_cuda_graph=true"])
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Union

import yaml


@dataclass
class ServerSection:
    host: str = "127.0.0.1"
    port: int = 8000
    rank_id: int = 0
    nccl_port: int = 28765
    tp: int = 1


@dataclass
class ModelSection:
    dir: Optional[str] = None
    tokenizer_mode: str = "slow"
    trust_remote_code: bool = False
    half_model: bool = False
    eos_id: int = 2
    mode: List[str] = field(default_factory=list)


@dataclass
class ServingSection:
    max_total_token_num: int = 6000
    max_req_input_len: int = 512
    max_req_total_len: int = 1024
    batch_max_tokens: Optional[int] = None
    running_max_req_size: int = 1000


@dataclass
class LoRASection:
    adapter_dirs: List[str] = field(default_factory=list)
    pool_size_lora: int = 0
    swap: bool = False
    prefetch: bool = False
    prefetch_size: int = 0
    batch_num_adapters: Optional[int] = None
    fair_weights: List[int] = field(default_factory=list)


@dataclass
class SchedulerSection:
    name: str = "dserve"
    enable_abort: bool = False
    # Path where the BatchExecutionTracker writes per-batch scheduling
    # decisions (predicted vs. actual durations, batch composition) at
    # finetuning exit. Path may be absolute or relative; relative paths
    # resolve against the active project root (eval/llama3/ in practice).
    # Parent directories are created on write. Set to null to disable.
    batch_prediction_stats_path: Optional[str] = "output/scheduler/batch_prediction_stats.csv"


@dataclass
class MemorySection:
    enable_unified_mem_manager: bool = True
    unified_mem_manager_max_size_gb: int = 6
    max_finetuning_tokens: int = 1024
    unified_mem_manager_log_path: Optional[str] = None
    # Allocator implementation:
    #   "auto"      — pick based on the model's GQA factor (default).
    #                 GQA (F > 1, e.g. Llama-3) → packed_kv; MHA (F == 1,
    #                 e.g. Llama-1/2) → unified, since packed_kv is then
    #                 byte-for-byte identical to unified anyway.
    #   "unified"   — page = 1 KV slot (legacy, GQA-unaware; wastes ~F× KV pool).
    #   "packed_kv" — page = F KV slots, F = num_attention_heads /
    #                 num_key_value_heads. Force this to exercise the
    #                 subclass on an MHA model for regression testing.
    allocator: str = "auto"


@dataclass
class CudaGraphSection:
    enable_decode_cuda_graph: bool = False
    enable_prefill_cuda_graph: bool = False
    enable_bwd_cuda_graph: bool = False
    use_graphed_bwd_attention: bool = True
    attn_bn_max: int = 8
    attn_l_max: int = 64
    # Upper bound on the offline prefill profiling sweep. Each unique
    # ceil(total/128)*128 bucket below this gets a captured prefill graph.
    # Lower this when GPU memory is tight; runtime batches above the cap
    # lazily capture on first hit (one-time latency spike). null = use
    # INF_CAP from the generator (i.e. batch_max_tokens / 2 of max_total).
    prefill_sweep_max_tokens: Optional[int] = None
    # Cap on total CUDA-graph memory (decode + prefill graphs combined).
    # When the live capture path would push total bytes over this, the
    # runner refuses to capture and falls back to eager forward for that
    # bucket. Once tripped, no further runtime captures happen for the
    # rest of the run (graph memory only grows). null = no cap.
    max_graph_memory_gb: Optional[float] = None


@dataclass
class FinetuneSection:
    enabled: bool = False
    type: str = "SFT"
    data_path: Optional[str] = None
    lora_path: Optional[str] = None
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gamma: float = 0.95
    max_saved_finetuning_tokens: int = 512
    max_finetuning_tokens_in_batch: int = 256
    optimizer_threading: bool = False
    start_on_launch: bool = True
    prepare_size: int = 999999
    log_path: str = ""


@dataclass
class SLOSection:
    ttft_slo: float = 0.35
    avg_tbt_slo: float = 0.15
    max_tbt_slo: float = 0.35


@dataclass
class DebugSection:
    dummy: bool = False
    no_lora: bool = False
    no_lora_compute: bool = False
    no_lora_swap: bool = False
    no_kernel: bool = False
    no_mem_pool: bool = False
    bmm: bool = False
    profile: bool = False
    enable_gpu_profile: bool = False
    disable_log_stats: bool = False


@dataclass
class ServerConfig:
    server: ServerSection = field(default_factory=ServerSection)
    model: ModelSection = field(default_factory=ModelSection)
    serving: ServingSection = field(default_factory=ServingSection)
    lora: LoRASection = field(default_factory=LoRASection)
    scheduler: SchedulerSection = field(default_factory=SchedulerSection)
    memory: MemorySection = field(default_factory=MemorySection)
    cuda_graph: CudaGraphSection = field(default_factory=CudaGraphSection)
    finetune: FinetuneSection = field(default_factory=FinetuneSection)
    slo: SLOSection = field(default_factory=SLOSection)
    debug: DebugSection = field(default_factory=DebugSection)

    def pretty(self) -> str:
        """Render as YAML — same shape as the input file, one field per line."""
        return yaml.safe_dump(
            asdict(self), default_flow_style=False, sort_keys=False, indent=2,
        ).rstrip()


_SECTION_CLASSES = {
    "server": ServerSection,
    "model": ModelSection,
    "serving": ServingSection,
    "lora": LoRASection,
    "scheduler": SchedulerSection,
    "memory": MemorySection,
    "cuda_graph": CudaGraphSection,
    "finetune": FinetuneSection,
    "slo": SLOSection,
    "debug": DebugSection,
}


def _section_from_dict(cls, data):
    if data is None:
        return cls()
    valid_names = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - valid_names
    if unknown:
        raise ValueError(
            f"unknown keys in section {cls.__name__}: {sorted(unknown)}. "
            f"Allowed: {sorted(valid_names)}"
        )
    return cls(**data)


def load_config(path: Union[str, Path]) -> ServerConfig:
    """Load a ServerConfig from a YAML file. Missing sections/fields get defaults."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"top-level YAML must be a mapping, got {type(raw).__name__}")

    unknown = set(raw.keys()) - set(_SECTION_CLASSES.keys())
    if unknown:
        raise ValueError(
            f"unknown sections in {path}: {sorted(unknown)}. "
            f"Allowed: {sorted(_SECTION_CLASSES.keys())}"
        )

    kwargs = {}
    for name, section_cls in _SECTION_CLASSES.items():
        if name in raw:
            kwargs[name] = _section_from_dict(section_cls, raw[name])
    return ServerConfig(**kwargs)


def apply_overrides(cfg: ServerConfig, overrides: List[str]) -> None:
    """
    Apply dotted-path overrides in place. Each override is `section.field=VALUE`;
    VALUE is YAML-parsed so booleans, ints, lists, and null are handled naturally.

    Examples:
        serving.max_total_token_num=30000
        cuda_graph.enable_bwd_cuda_graph=true
        lora.adapter_dirs=[a,b,c]
        finetune.lora_path=null
    """
    for o in overrides or []:
        if "=" not in o:
            raise ValueError(f"override must be KEY=VALUE, got {o!r}")
        key, _, val_str = o.partition("=")
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"override key must be section.field, got {key!r}")
        section_name, field_name = parts
        if not hasattr(cfg, section_name):
            raise ValueError(f"unknown section {section_name!r} in override {o!r}")
        section = getattr(cfg, section_name)
        valid_names = {f.name for f in fields(type(section))}
        if field_name not in valid_names:
            raise ValueError(
                f"unknown field {section_name}.{field_name} in override {o!r}. "
                f"Allowed: {sorted(valid_names)}"
            )
        setattr(section, field_name, yaml.safe_load(val_str))
