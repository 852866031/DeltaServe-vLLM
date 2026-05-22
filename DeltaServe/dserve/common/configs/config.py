"""
Process-wide active ServerConfig accessor.

Replaces the previous global `setting` dict. Each subprocess that needs
config reads must call `set_active_config()` once at startup before any
hot-path code calls `get_active_config()`.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dserve.server.config import ServerConfig

_cfg: Optional["ServerConfig"] = None


def set_active_config(cfg: "ServerConfig") -> None:
    global _cfg
    _cfg = cfg


def get_active_config() -> "ServerConfig":
    if _cfg is None:
        raise RuntimeError(
            "active ServerConfig not initialized — call set_active_config() at "
            "subprocess startup before any infer-batch / model-rpc work"
        )
    return _cfg
