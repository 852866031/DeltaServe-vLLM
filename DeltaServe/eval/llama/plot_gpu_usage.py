import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os

def plot_gpu_usage(csv_path, output_png="gpu_usage.png"):
    # === Load CSV ===
    df = pd.read_csv(csv_path)

    # Parse timestamps → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute timeline relative to first timestamp
    t0 = df["timestamp"].min()
    df["time_rel_s"] = (df["timestamp"] - t0).dt.total_seconds()

    # Determine GPUs present
    gpu_ids = sorted(df["gpu_index"].unique())
    num_gpus = len(gpu_ids)

    # === Plot ===
    fig, axes = plt.subplots(num_gpus, 1, figsize=(12, 3 * num_gpus), sharex=True)

    # If only 1 GPU, axes is not a list
    if num_gpus == 1:
        axes = [axes]

    for ax, gpu_id in zip(axes, gpu_ids):
        d = df[df["gpu_index"] == gpu_id]

        ax.plot(d["time_rel_s"], d["gpu_util"], linewidth=1.2)
        ax.set_ylabel(f"GPU {gpu_id} Util (%)")
        ax.set_ylim(-5, 110)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle("GPU Utilization Over Time", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_png)
    print(f"✅ Saved plot to {output_png}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = current_dir + "/results/gpu_usage_multi_co.csv"
    out = current_dir + "/gpu_usage.png"
    plot_gpu_usage(csv_file, out)