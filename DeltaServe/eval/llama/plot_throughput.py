#!/usr/bin/env python3
"""
Plot throughput timeline from <stem>_throughput.csv.

Input format:
    second,total_tokens

Usage:
    python plot_throughput.py throughput.csv --out plot.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_throughput(csv_path, out_path=None, bar=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"second", "total_tokens"} <= set(df.columns):
        raise ValueError("CSV must contain columns: second,total_tokens")

    df["second"] = pd.to_numeric(df["second"], errors="coerce")
    df["total_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce")

    # Sort by time just in case
    df = df.sort_values("second")

    plt.figure(figsize=(16, 5))

    if bar:
        plt.bar(df["second"], df["total_tokens"], width=0.9, alpha=0.7)
    else:
        plt.plot(df["second"], df["total_tokens"], marker="", linewidth=1.5)

    plt.title("Inference Throughput (Tokens per Second)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Total tokens generated per second")

    plt.grid(True, linestyle="--", alpha=0.35)

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[plotter] Saved → {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Throughput CSV produced by write_throughput_csv()")
    ap.add_argument("--out", help="Output PNG path (optional)")
    ap.add_argument("--bar", action="store_true", help="Use bar style instead of line")
    args = ap.parse_args()

    plot_throughput(args.csv_path, args.out, bar=args.bar)


if __name__ == "__main__":
    main()