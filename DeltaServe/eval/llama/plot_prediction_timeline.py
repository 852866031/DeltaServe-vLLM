import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Helper: infer device name for title
# ------------------------------------------------------------
def pretty_title_from_path(path: str) -> str:
    name = path.lower()
    if "5090" in name:
        return "RTX 5090"
    if "4090" in name:
        return "RTX 4090"
    if "a100" in name:
        return "NVIDIA A100"
    if "h100" in name:
        return "NVIDIA H100"
    return path


# ------------------------------------------------------------
# CSV loader for the first two plots
# ------------------------------------------------------------
def _load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["inference_tokens"]  = df["inference_tokens"].apply(json.loads)
    df["finetuning_tokens"] = df["finetuning_tokens"].apply(json.loads)
    df["layout"] = df.apply(
        lambda r: (len(r["inference_tokens"]), len(r["finetuning_tokens"])),
        axis=1
    )
    return df


def _extract_aligned_arrays(df, layouts, layout_to_x):
    num_points = len(layouts)

    prefill_actual    = np.full(num_points, np.nan)
    prefill_predicted = np.full(num_points, np.nan)
    decode_actual     = np.full(num_points, np.nan)
    decode_predicted  = np.full(num_points, np.nan)

    for _, row in df.iterrows():
        layout = row["layout"]
        if layout not in layout_to_x:
            continue
        x = layout_to_x[layout]

        if row["batch_type"] == "prefill":
            prefill_actual[x]    = row["execution_duration"]
            prefill_predicted[x] = row["predicted_duration"]
        else:
            decode_actual[x]     = row["execution_duration"]
            decode_predicted[x]  = row["predicted_duration"]

    decode_x = np.arange(num_points)

    d_idx_actual = ~np.isnan(decode_actual)
    d_idx_pred   = ~np.isnan(decode_predicted)

    decode_actual_x = decode_x[d_idx_actual]
    decode_actual_y = decode_actual[d_idx_actual]

    decode_pred_x = decode_x[d_idx_pred]
    decode_pred_y = decode_predicted[d_idx_pred]

    return (
        prefill_actual, prefill_predicted,
        decode_actual, decode_predicted,
        decode_actual_x, decode_actual_y,
        decode_pred_x, decode_pred_y
    )


# ------------------------------------------------------------
# THIRD PLOT: pick the **third 30-second window**
# ------------------------------------------------------------
def extract_third_30sec_window(df):
    df = df[df["batch_type"] == "prefill"].copy()
    if df.empty:
        return pd.DataFrame()

    df["win30"] = (df["timestamp"] // 30).astype(int)
    wins = sorted(df["win30"].unique())

    if len(wins) < 3:
        return pd.DataFrame()

    target = wins[2]    # 3rd window

    out = df[df["win30"] == target].copy()
    if out.empty:
        return out

    base = out["timestamp"].min()
    out["timestamp"] -= base

    return out


# ------------------------------------------------------------
# Main three-panel plot
# ------------------------------------------------------------
def plot_three_panel_with_timeline(csv_1, csv_2, timeline_csv, out_path="three_panel_stats.png"):
    df1 = _load_csv(csv_1)
    df2 = _load_csv(csv_2)
    df_t = pd.read_csv(timeline_csv)

    layouts = sorted(df1["layout"].unique())
    layout_to_x = {layout: i for i, layout in enumerate(layouts)}

    arr1 = _extract_aligned_arrays(df1, layouts, layout_to_x)
    arr2 = _extract_aligned_arrays(df2, layouts, layout_to_x)

    df_timeline = extract_third_30sec_window(df_t)

    title1 = pretty_title_from_path(csv_1)
    title2 = pretty_title_from_path(csv_2)
    title3 = "Company X Workload Prefill Duration"

    c_actual = "tab:blue"
    c_pred   = "tab:orange"

    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5))
    global_handles = []
    global_labels  = []

    # ------------------------------------------------------------
    # Subplots 1 and 2
    # ------------------------------------------------------------
    arrays = [arr1, arr2]
    titles = [title1, title2]

    for ax, arr, title in zip(axes[:2], arrays, titles):
        (prefill_actual, prefill_predicted,
         decode_actual, decode_predicted,
         decode_actual_x, decode_actual_y,
         decode_pred_x, decode_pred_y) = arr

        # Prefill
        h_preal, = ax.plot(prefill_actual,    marker="o", linestyle="-",  color=c_actual)
        h_prpre, = ax.plot(prefill_predicted, marker="o", linestyle="--", color=c_pred)

        # Decode
        h_deal, = ax.plot(decode_actual_x, decode_actual_y, marker="s", linestyle="-",  color=c_actual)
        h_depre, = ax.plot(decode_pred_x,   decode_pred_y,   marker="s", linestyle="--", color=c_pred)

        if not global_handles:
            global_handles = [h_preal, h_prpre, h_deal, h_depre]
            global_labels  = [
                "Prefill – Actual",
                "Prefill – Predicted",
                "Decode – Actual",
                "Decode – Predicted"
            ]

        # Layout labels on x-axis
        layout_labels = [f"({inf},{ft})" for (inf, ft) in layouts]
        ax.set_xticks(np.arange(len(layouts)))
        ax.set_xticklabels(layout_labels, rotation=45, fontsize=10)

        # Error % annotation
        prefill_err_pct = np.nanmean(np.abs(prefill_predicted - prefill_actual) / prefill_actual * 100)
        decode_err_diff = np.nanmean(np.abs(decode_predicted - decode_actual))  # in seconds

        ax.text(
            0.02, 0.95,
            f"Prefill Err: {prefill_err_pct:.2f}%\nDecode Err: +{decode_err_diff:.3f}s",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
        )

        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Duration (s)", fontsize=12)
        ax.set_xlabel("Batch Layout (#inf_reqs, #ft_samples)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.35)

    # ------------------------------------------------------------
    # Subplot 3 — timeline on **third 30s window**
    # ------------------------------------------------------------
    ax = axes[2]

    if df_timeline.empty:
        ax.text(0.5, 0.5, "No prefill rows in 3rd 30s window", ha="center")
    else:
        ax.plot(df_timeline["timestamp"], df_timeline["execution_duration"],
                label="Actual", color=c_actual, linewidth=1.2)
        ax.plot(df_timeline["timestamp"], df_timeline["predicted_duration"],
                label="Predicted", linestyle="--", color=c_pred, linewidth=1.2)

        # --- Percent annotation ---
        abs_err = np.abs(df_timeline["execution_duration"] - df_timeline["predicted_duration"])
        pct_err = (abs_err / df_timeline["execution_duration"]) * 100
        mean_pct = pct_err.mean()

        over_pred = (df_timeline["predicted_duration"] > df_timeline["execution_duration"]).mean() * 100

        ax.text(
            0.02, 0.95,
            f"Avg Err: {mean_pct:.2f}%",
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8)
        )

    ax.set_title(title3, fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Duration (s)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_yticks([0.02, 0.03])
    ax.set_xticks([0, 10, 20, 30])

    # ------------------------------------------------------------
    # GLOBAL LEGEND
    # ------------------------------------------------------------
    fig.legend(
        handles=global_handles,
        labels=global_labels,
        loc="upper center",
        ncol=2,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0)
    print(f"Saved PDF -> {out_path}")


# ------------------------------------------------------------
# Example run
# ------------------------------------------------------------
if __name__ == "__main__":
    plot_three_panel_with_timeline(
        "batch_prediction_stats_5090.csv",
        "batch_prediction_stats_A100.csv",
        "batch_prediction_stats.csv",
        out_path="profiling.pdf"
    )