#!/usr/bin/env python3
"""
Plot latency timeline from merged multi-GPU benchmark CSV.
X-axis: t_rel_s (request relative send time)
Y-axis: latency_s (total request latency)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_latency(csv_path, out_path=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    for col in ["t_rel_s", "latency_s", "status"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    # Convert types safely
    df["t_rel_s"] = pd.to_numeric(df["t_rel_s"], errors="coerce")
    df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")

    # Keep only successful rows
    df_ok = df[df["status"] == "ok"].dropna(subset=["t_rel_s", "latency_s"])

    plt.figure(figsize=(14, 5))
    plt.scatter(df_ok["t_rel_s"], df_ok["latency_s"], s=8, alpha=0.6)

    plt.title("Latency vs Relative Send Time")
    plt.xlabel("t_rel_s (seconds)")
    plt.ylabel("latency_s (seconds)")

    plt.grid(True, linestyle="--", alpha=0.3)

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot → {out_path}")


def main():
    csv_path = "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/results/latency_co-serving.csv"
    out_png = "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/results/latency_co-serving.png"
    plot_latency(csv_path, out_png)
    csv_path = "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/results/latency_slora.csv"
    out_png = "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/results/latency_slora.png"
    plot_latency(csv_path, out_png)



if __name__ == "__main__":
    main()