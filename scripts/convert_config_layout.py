#!/usr/bin/env python
"""Rewrite a DeltaServe serving YAML into the sectioned finetune layout
(finetune / throttles / slo / debug), keeping every comment next to its key.

    python scripts/convert_config_layout.py configs/old.yaml            # print
    python scripts/convert_config_layout.py configs/old.yaml -o new.yaml
    python scripts/convert_config_layout.py --in-place configs/*.yaml

Line-based on purpose (no YAML round-trip, which would drop the comments): a key
line moves to its section together with the comment lines directly above it.
Retired keys are dropped. Non-finetune sections are copied verbatim. The result
is checked to load to the same FinetuneConfig kwargs and engine sections.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "deltaserve-plugin"))


def _sections_table():
    # Read the tables from config_loader's source (importing it would pull in vLLM).
    import ast
    src = (_ROOT / "deltaserve-plugin/deltaserve/config_loader.py").read_text()
    tables = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id in ("FINETUNE_SECTIONS", "RETIRED_KEYS"):
            tables[node.target.id] = ast.literal_eval(node.value)
    return tables["FINETUNE_SECTIONS"], tables["RETIRED_KEYS"]


FINETUNE_SECTIONS, RETIRED_KEYS = _sections_table()
SECTION_OF = {k: s for s, keys in FINETUNE_SECTIONS.items() for k in keys}
_SECTION_RE = re.compile(r"^([A-Za-z_][\w-]*):\s*(#.*)?$")
_KEY_RE = re.compile(r"^\s+([A-Za-z_][\w-]*):")


def convert(text: str) -> str:
    lines = text.splitlines()
    preamble: list[str] = []
    order: list[str] = []                    # top-level sections, file order
    body: dict[str, list[str]] = {}          # non-finetune sections, verbatim
    header: dict[str, str] = {}              # section header line (with comment)
    items: dict[str, list[list[str]]] = {s: [] for s in FINETUNE_SECTIONS}
    tails: dict[str, list[str]] = {s: [] for s in FINETUNE_SECTIONS}
    cur = None
    pending: list[str] = []                  # comment lines awaiting their key
    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            if cur in FINETUNE_SECTIONS:
                tails[cur] += pending
            pending = []
            cur = m.group(1)
            order.append(cur)
            header[cur] = line
            body.setdefault(cur, [])
            continue
        if cur is None:
            preamble.append(line)
            continue
        if cur not in FINETUNE_SECTIONS:
            body[cur].append(line)
            continue
        if not line.strip():
            pending.append(line)
            continue
        if line.lstrip().startswith("#"):
            pending.append(line)
            continue
        km = _KEY_RE.match(line)
        key = km.group(1) if km else None
        block = [ln for ln in pending if ln.strip()] + [line]
        pending = []
        if key in RETIRED_KEYS:
            continue
        items[SECTION_OF.get(key, cur)].append(block)
    if cur in FINETUNE_SECTIONS:
        tails[cur] += pending

    out = list(preamble)
    emitted_ft = False
    for name in order:
        if name in FINETUNE_SECTIONS:
            if emitted_ft:
                continue
            emitted_ft = True
            for sec in FINETUNE_SECTIONS:
                if not items[sec] and not any(t.strip() for t in tails[sec]):
                    continue
                if out and out[-1].strip():
                    out.append("")
                out.append(header.get(sec, f"{sec}:"))
                for block in items[sec]:
                    out += block
                out += [t for t in tails[sec] if t.strip()]
            continue
        if out and out[-1].strip():
            out.append("")
        out.append(header[name])
        out += body[name]
        while out and not out[-1].strip():
            out.pop()
    return "\n".join(out).rstrip() + "\n"


def _check(old: str, new: str) -> None:
    a, b = yaml.safe_load(old) or {}, yaml.safe_load(new) or {}

    def ft(cfg):
        out = {}
        for sec in FINETUNE_SECTIONS:
            for k, v in (cfg.get(sec) or {}).items():
                if k not in RETIRED_KEYS:
                    out[k] = v
        return out

    assert ft(a) == ft(b), "finetune settings changed"
    rest = lambda c: {k: v for k, v in c.items() if k not in FINETUNE_SECTIONS}  # noqa: E731
    assert rest(a) == rest(b), "non-finetune sections changed"
    for sec in FINETUNE_SECTIONS:
        for k in (b.get(sec) or {}):
            assert SECTION_OF.get(k, sec) == sec, f"{k} left in {sec}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--output")
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()
    if args.output and len(args.paths) != 1:
        ap.error("-o takes a single input")
    for p in args.paths:
        old = Path(p).read_text()
        new = convert(old)
        _check(old, new)
        if args.in_place:
            Path(p).write_text(new)
            print(f"converted {p}")
        elif args.output:
            Path(args.output).write_text(new)
        else:
            sys.stdout.write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
