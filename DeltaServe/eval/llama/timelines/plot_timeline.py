import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # === Determine input CSV path ===
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)

    # If a filename is passed as a command-line argument, use it
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        csv_filename = "timeline_live.csv"

    # Support both absolute and relative paths
    if not os.path.isabs(csv_filename):
        csv_path = os.path.join(current_directory, csv_filename)
    else:
        csv_path = csv_filename

    if not os.path.exists(csv_path):
        print(f"❌ Error: file '{csv_path}' not found.")
        sys.exit(1)

    print(f"📂 Using input file: {csv_path}")

    # === Load CSV ===
    df = pd.read_csv(csv_path)

    # === Compute total tokens per request ===
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]

    # === Manually compute "second" bins ===
    # floor the timestamp to integer seconds: 1.0–<2.0 → second 1, etc.
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)

    # === Aggregate manually by the computed bins ===
    tokens_per_sec = df.groupby("second_bin")["total_tokens"].sum().rename("total_tokens_per_sec")
    req_per_sec = df.groupby("second_bin").size().rename("num_requests")
    timeline = pd.concat([tokens_per_sec, req_per_sec], axis=1).reset_index()

    all_seconds = pd.RangeIndex(
        start=timeline["second_bin"].min(),
        stop=timeline["second_bin"].max() + 1
    )
    timeline = timeline.set_index("second_bin").reindex(all_seconds, fill_value=0).rename_axis("second_bin").reset_index()
    # === Plot ===
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left Y-axis: total tokens per second
    ax1.plot(
        timeline["second_bin"],
        timeline["total_tokens_per_sec"],
        color="tab:blue",
        linewidth=2,
        label="Total tokens/sec"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Total tokens per second", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(top=timeline["total_tokens_per_sec"].max() * 1.3, bottom=0)
    ax1.set_xlim(left=0)

    # Right Y-axis: number of requests per second
    ax2 = ax1.twinx()
    ax2.bar(
        timeline["second_bin"],
        timeline["num_requests"],
        color="tab:orange",
        alpha=0.4,
        width=0.9,
        label="#Requests/sec"
    )
    ax2.set_ylabel("Number of requests", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(top=timeline["num_requests"].max() * 1.3, bottom=0)
    ax2.set_xlim(left=0)

    # === Aesthetics ===
    fig.suptitle("Inference Request Timeline (Manual Per-Second Binning)")
    
    fig.tight_layout()

    # === Save figure ===
    output_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(output_path)
    print(f"✅ Saved plot to {output_path}")

if __name__ == "__main__":
    main()