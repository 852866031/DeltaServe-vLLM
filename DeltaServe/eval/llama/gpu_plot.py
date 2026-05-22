import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_gpu_compute_usage_only(
    file1, label1,
    file2, label2,
    file3=None, label3=None,
    out_png="gpu_compute_usage_only.png",
    smooth_window: int = 1,
):
    """
    Compare 2 or 3 GPU usage log files (only GPU utilization, no memory).

    - Uses relative time (seconds since start of each log)
    - Smooths short-term noise with rolling average
    - GPU curves appear subtle, with bold average utilization lines
    """

    # ---------------- Load helper ---------------- #
    def load_log(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        # Drop the first 10 seconds (startup noise)
        df = df[df["time_rel_s"] >= 10].reset_index(drop=True)
        if "gpu_util" not in df.columns:
            raise ValueError(f"{path} missing required column 'gpu_util'")
        df["gpu_util_smooth"] = df["gpu_util"].rolling(window=smooth_window, min_periods=1).mean()
        return df

    df1 = load_log(file1)
    df2 = load_log(file2)
    df3 = load_log(file3) if file3 else None

    # ---------------- Compute averages ---------------- #
    avg1 = df1["gpu_util"].mean()
    avg2 = df2["gpu_util"].mean()
    avg3 = df3["gpu_util"].mean() if df3 is not None else None

    # ---------------- Create plot ---------------- #
    fig, ax = plt.subplots(figsize=(12, 5))

    # --- Colors ---
    color_map = ["tab:blue", "tab:orange", "tab:purple"]

    # --- GPU Utilization Curves (less visible) ---
    ax.plot(df1["time_rel_s"], df1["gpu_util_smooth"], color=color_map[0], linewidth=1.0, alpha=0.55, label=f"{label1} GPU Util (curve)")
    ax.plot(df2["time_rel_s"], df2["gpu_util_smooth"], color=color_map[1], linewidth=1.0, alpha=0.75, label=f"{label2} GPU Util (curve)")
    if df3 is not None:
        ax.plot(df3["time_rel_s"], df3["gpu_util_smooth"], color=color_map[2], linewidth=1.0, alpha=0.55, label=f"{label3} GPU Util (curve)")

    # --- Bold Average Lines ---
    ax.axhline(avg1, color=color_map[0], linestyle="--", linewidth=2.0, alpha=0.9, label=f"{label1} avg={avg1:.1f}%")
    ax.axhline(avg2, color=color_map[1], linestyle="--", linewidth=2.0, alpha=0.9, label=f"{label2} avg={avg2:.1f}%")
    if df3 is not None:
        ax.axhline(avg3, color=color_map[2], linestyle="--", linewidth=2.0, alpha=0.9, label=f"{label3} avg={avg3:.1f}%")

    # --- Labels and Style ---
    ax.set_xlabel("Time (s, relative)", fontsize=11)
    ax.set_ylabel("GPU Utilization (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    title = f"GPU Compute Utilization (Relative Time): {label1} vs {label2}"
    if df3 is not None:
        title += f" vs {label3}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    fig.tight_layout()
    out_path = os.path.abspath(out_png)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"✅ Saved compute-only GPU usage plot to {out_path}")
    print(f"Average Utilizations — {label1}: {avg1:.1f}%, {label2}: {avg2:.1f}%" + (f", {label3}: {avg3:.1f}%" if df3 is not None else ""))


# ---------------- CLI entrypoint ---------------- #
if __name__ == "__main__":
    file1 = 'results/gpu_usage_inference.csv'
    file2 = 'results/gpu_usage_co-serving.csv'
    label1 = 'inf'
    label2 = 'co_serving'
    file3 = 'results/gpu_usage_slora.csv'
    label3 = 'SLoRA'

    plot_gpu_compute_usage_only(file1, label1, file2, label2, file3, label3)