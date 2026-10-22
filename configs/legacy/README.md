# Legacy-layout configs

The serving YAMLs as they were before the 2026-10 section restructure (finetune
settings spread over `finetune` / `slo` / `debug`). Relative paths were adjusted
by one directory level so these still load. The loader accepts this layout with
a warning per misplaced key; `scripts/convert_config_layout.py` rewrites a file
into the current layout (`finetune` / `throttles` / `slo` / `debug`), keeping its
comments.
