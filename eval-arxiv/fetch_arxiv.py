#!/usr/bin/env python
"""Fetch the arxiv-summarization test split and keep the longest articles.

The HuggingFace dataset ``ccdv/arxiv-summarization`` has two columns,
``article`` (full paper body) and ``abstract``. This pulls ONE parquet shard
of the ``document`` config's test split (6440 papers, ~104 MB) straight from
the Hub's parquet endpoint (no `datasets` library needed) and writes the N
longest articles (by character count) to a JSONL the probe / trace builder
consume: ``{"idx", "article", "abstract"}`` per line.

Needs ``pyarrow`` — the dserve-vllm env does not ship it, the base miniconda
python does:

    /home/jiaxuan/miniconda3/bin/python eval-arxiv/fetch_arxiv.py --out eval-arxiv/data/long_articles.jsonl
"""
import argparse, json, os, statistics as st, sys, urllib.request

URL = ("https://huggingface.co/api/datasets/ccdv/arxiv-summarization/parquet/"
       "{config}/{split}/0.parquet")

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="document", choices=["document", "section"])
ap.add_argument("--split", default="test")
ap.add_argument("--keep", type=int, default=400, help="how many longest articles to keep")
ap.add_argument("--parquet", default=None, help="reuse an already-downloaded shard")
ap.add_argument("--out", required=True)
args = ap.parse_args()

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow missing — run with a python that has it "
             "(e.g. /home/jiaxuan/miniconda3/bin/python)")

pq_path = args.parquet or os.path.join(os.path.dirname(args.out) or ".",
                                       f"arxiv_{args.config}_{args.split}_0.parquet")
os.makedirs(os.path.dirname(pq_path) or ".", exist_ok=True)
if not os.path.exists(pq_path):
    url = URL.format(config=args.config, split=args.split)
    print(f"[fetch] downloading {url} -> {pq_path}", flush=True)
    urllib.request.urlretrieve(url, pq_path)
t = pq.read_table(pq_path)
arts = t.column("article").to_pylist(); abss = t.column("abstract").to_pylist()
print(f"[fetch] {t.num_rows} rows; article chars p50={st.median(len(a) for a in arts):.0f}", flush=True)
order = sorted(range(len(arts)), key=lambda i: len(arts[i]), reverse=True)[:args.keep]
with open(args.out, "w") as f:
    for i in order:
        f.write(json.dumps({"idx": i, "article": arts[i], "abstract": abss[i]}) + "\n")
print(f"[fetch] wrote {len(order)} articles to {args.out} "
      f"(shortest kept: {len(arts[order[-1]])} chars)", flush=True)
