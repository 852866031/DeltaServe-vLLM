import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
result_dir = os.path.dirname(os.path.abspath(__file__))

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
case_labels  = ["DeltaServe", "DeltaServe-Inf", "S-LoRA"]

colors = {
    "DeltaServe":      "#ff7f0e",
    "DeltaServe-Inf":  "#2ca02c",
    "S-LoRA":           "#1f77b4",
}

# Finetuning throughput (tokens/sec)
ft_throughput = [1792/8, 512/8, 256/8]

TTFT_SLO = 0.10
missing_hatch = "//"

# Only DeltaServe-Inf has hatch
special_hatch = {"DeltaServe-Inf": "//"}

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

fig, (ax_ttft, ax_lat) = plt.subplots(1, 2, figsize=(8, 2.5), layout='constrained')

x = np.arange(num_groups)
bar_width = 0.18
spacing_factor = 1.2
offsets = spacing_factor * np.linspace(-bar_width, bar_width, num_cases)


# ====================== LEFT PANEL — TTFT SLO ======================
for c_idx in range(num_cases):
    label = case_labels[c_idx]
    hatch = special_hatch.get(label, "")

    for g_idx in range(num_groups):
        exists = exists_mask[g_idx, c_idx]

        # DeltaServe-Inf gets green fill + black hatch
        if hatch and exists:
            fc = colors[label]
            ec = "black"
        else:
            fc = colors[label] if exists else "white"
            ec = colors[label]

        ax_ttft.bar(
            x[g_idx] + offsets[c_idx],
            ttft_slo_pct[g_idx, c_idx],
            width=bar_width,
            facecolor=fc,
            edgecolor=ec,
            hatch=hatch if exists else missing_hatch,
            linewidth=1.4,
        )

ax_ttft.set_title("P99 SLO Satisfaction", fontsize=15, pad=1)
ax_ttft.set_xticks(x)
ax_ttft.set_xticklabels(group_labels)
ax_ttft.set_ylabel("SLO Satisfaction (%)")
ax_ttft.set_ylim(0, 105)


# ====================== RIGHT PANEL — LATENCY + FT Throughput ======================
for c_idx in range(num_cases):
    label = case_labels[c_idx]
    hatch = special_hatch.get(label, "")

    for g_idx in range(num_groups):
        exists = exists_mask[g_idx, c_idx]

        if hatch and exists:
            fc = colors[label]
            ec = "black"
        else:
            fc = colors[label] if exists else "white"
            ec = colors[label]

        ax_lat.bar(
            x[g_idx] + offsets[c_idx],
            avg_latency[g_idx, c_idx],
            width=bar_width,
            facecolor=fc,
            edgecolor=ec,
            hatch=hatch if exists else missing_hatch,
            linewidth=1.4,
        )

ax_lat.set_title("Average End-to-End Latency", fontsize=15, pad=1)
ax_lat.set_xticks(x)
ax_lat.set_xticklabels(group_labels)
ax_lat.set_ylabel("Seconds")

# FT throughput line
ax_lat2 = ax_lat.twinx()
ax_lat2.plot(x, ft_throughput, marker="o", linestyle="-", color="black", linewidth=2)
ax_lat2.set_ylabel("FT Throughput (tokens/sec)")
ax_lat2.set_ylim(0, max(ft_throughput) * 1.2)


# ====================== LEGEND ======================
legend_handles = []
for label in case_labels:
    hatch = special_hatch.get(label, "")

    if hatch:  # DeltaServe-Inf
        fc = colors[label]
        ec = "black"
    else:
        fc = colors[label]
        ec = colors[label]

    legend_handles.append(
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=fc,
            edgecolor=ec,
            hatch=hatch,
            linewidth=1.2,
        )
    )

fig.legend(
    handles=legend_handles,
    labels=case_labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=15,
    bbox_to_anchor=(0.5, 1.2),
    bbox_transform=fig.transFigure
)


# ====================== SAVE ======================
out_file = "ablation1_results.pdf"
print(f"Saving figure to {out_file}")
plt.savefig(out_file, dpi=300, format="pdf", bbox_inches="tight", pad_inches=0)