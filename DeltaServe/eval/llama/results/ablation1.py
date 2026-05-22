import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
result_dir = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/eval/results"

group_files = [
    [f"{result_dir}/latency_co-serving_4rps.csv",
     f"{result_dir}/latency_inference_4rps.csv",
     f"{result_dir}/latency_slora_4rps.csv"],

    [f"{result_dir}/latency_co-serving_8rps.csv",
     f"{result_dir}/latency_inference_8rps.csv",
     f"{result_dir}/latency_slora_8rps.csv"],

    [f"{result_dir}/latency_co-serving_12rps.csv",
     f"{result_dir}/latency_inference_12rps.csv",
     f"{result_dir}/latency_slora_12rps.csv"],
]

group_labels = ["RPS 4", "RPS 8", "RPS 12"]
case_labels  = ["Co", "Inf", "SLoRA"]

colors = {
    "Co":    "#ff7f0e",  # orange
    "Inf":   "#2ca02c",  # green
    "SLoRA": "#1f77b4",  # blue
}

# NEW: Finetuning throughput tokens/sec for each group
ft_throughput = [1792/8, 512/8, 256/8]  # user-provided list

TTFT_SLO = 0.10
missing_hatch = "//"


# ---------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------
def summarize_file(path: str, ttft_slo: float):
    if not os.path.exists(path):
        return 0.0, 0.0, False
    try:
        df = pd.read_csv(path)
    except:
        return 0.0, 0.0, False

    df = df[df["status"] == "ok"]
    if len(df) == 0:
        return 0.0, 0.0, False

    avg_total_latency = df["latency_s"].mean()

    sorted_ttft = np.sort(df["ttft_s"].to_numpy())
    cutoff_idx = int(len(sorted_ttft) * 0.99)
    cutoff = sorted_ttft[:cutoff_idx] if cutoff_idx > 0 else sorted_ttft
    pct_fast_satisfied = 100.0 * np.mean(cutoff <= ttft_slo) if len(cutoff) > 0 else 0.0

    return avg_total_latency, pct_fast_satisfied, True


# ---------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------
num_groups = len(group_files)
num_cases = len(group_files[0])

avg_latency = np.zeros((num_groups, num_cases))
ttft_slo_pct = np.zeros((num_groups, num_cases))
exists_mask = np.zeros((num_groups, num_cases), dtype=bool)

for g_idx, files in enumerate(group_files):
    for c_idx, path in enumerate(files):
        lat, pct, exists = summarize_file(path, TTFT_SLO)
        avg_latency[g_idx, c_idx] = lat
        ttft_slo_pct[g_idx, c_idx] = pct
        exists_mask[g_idx, c_idx] = exists


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
plt.rcParams["font.size"] = 13
plt.rcParams["axes.linewidth"] = 1.2

# NOTE: REVERSED ORDER NOW (TTFT on left, latency on right)
fig, (ax_ttft, ax_lat) = plt.subplots(
    1, 2, figsize=(8, 3), layout='constrained'
)

x = np.arange(num_groups)
bar_width = 0.18
spacing_factor = 1.2
offsets = spacing_factor * np.linspace(-bar_width, bar_width, num_cases)


# ====================== LEFT PANEL — 99% TTFT SLO ======================
for c_idx in range(num_cases):
    label = case_labels[c_idx]
    for g_idx in range(num_groups):
        exists = exists_mask[g_idx, c_idx]
        ax_ttft.bar(
            x[g_idx] + offsets[c_idx],
            ttft_slo_pct[g_idx, c_idx],
            width=bar_width,
            color=colors[label] if exists else "white",
            edgecolor=colors[label],
            hatch=missing_hatch if not exists else "",
            linewidth=1.4,
        )

ax_ttft.set_title("99% TTFT SLO Satisfaction", fontsize=16, pad=12)
ax_ttft.set_xticks(x)
ax_ttft.set_xticklabels(group_labels)
ax_ttft.set_ylabel("SLO Satisfaction (%)")
ax_ttft.set_ylim(0, 105)


# ====================== RIGHT PANEL — Avg Latency + FT Throughput ======
# Bars (latency)
for c_idx in range(num_cases):
    label = case_labels[c_idx]
    for g_idx in range(num_groups):
        exists = exists_mask[g_idx, c_idx]
        ax_lat.bar(
            x[g_idx] + offsets[c_idx],
            avg_latency[g_idx, c_idx],
            width=bar_width,
            color=colors[label] if exists else "white",
            edgecolor=colors[label],
            hatch=missing_hatch if not exists else "",
            linewidth=1.4,
        )

ax_lat.set_title("Average End-to-End Latency", fontsize=16, pad=12)
ax_lat.set_xticks(x)
ax_lat.set_xticklabels(group_labels)
ax_lat.set_ylabel("Seconds")

# Add second y-axis for FT throughput
ax_lat2 = ax_lat.twinx()
ax_lat2.plot(
    x,
    ft_throughput,
    marker="o",
    linestyle="-",
    color="black",
    linewidth=2,
)
ax_lat2.set_ylabel("FT Throughput (tokens/sec)", fontsize=12)
ax_lat2.tick_params(axis='y')
ax_lat2.set_ylim(0, max(ft_throughput)*1.2)


# ====================== Global Legend ======================
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[label], label=label)
    for label in case_labels
]

fig.legend(
    handles=legend_handles,
    labels=case_labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=14,
    bbox_to_anchor=(0.5, 1.18),
    bbox_transform=fig.transFigure
)

# ---------------------------------------------------------------------
out_file = "ablation1_results.png"
print(f"Saving figure to {out_file}")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.show()