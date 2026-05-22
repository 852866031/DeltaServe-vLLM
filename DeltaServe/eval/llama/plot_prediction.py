import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Helper: turn filename into a nice title
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
    return path  # fallback


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

    # Build continuous decode lines
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


def plot_two_files(csv_path_1, csv_path_2, out_path="compare_two_models.png"):
    df1 = _load_csv(csv_path_1)
    df2 = _load_csv(csv_path_2)

    # Shared layouts
    layouts = sorted(df1["layout"].unique())
    layout_to_x = {layout: i for i, layout in enumerate(layouts)}
    num_points = len(layouts)

    arr1 = _extract_aligned_arrays(df1, layouts, layout_to_x)
    arr2 = _extract_aligned_arrays(df2, layouts, layout_to_x)

    title1 = pretty_title_from_path(csv_path_1)
    title2 = pretty_title_from_path(csv_path_2)

    c_actual    = "tab:blue"
    c_predicted = "tab:orange"

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    arrays = [arr1, arr2]
    titles = [title1, title2]

    # ------------------------------------------------------------
    # Collect *all four* handles for global legend
    # ------------------------------------------------------------
    global_handles = []
    global_labels  = []

    for ax, arr, title in zip(axes, arrays, titles):
        (prefill_actual, prefill_predicted,
         decode_actual, decode_predicted,
         decode_actual_x, decode_actual_y,
         decode_pred_x, decode_pred_y) = arr

        # Prefill (circle markers)
        h_preal, = ax.plot(prefill_actual,    marker="o", linestyle="-",  color=c_actual,   alpha=0.8)
        h_prpre, = ax.plot(prefill_predicted, marker="o", linestyle="--", color=c_predicted, alpha=0.8)

        # Decode (square markers)
        h_deal, = ax.plot(decode_actual_x, decode_actual_y, marker="s", linestyle="-",  color=c_actual)
        h_depre, = ax.plot(decode_pred_x,   decode_pred_y,   marker="s", linestyle="--", color=c_predicted)

        # Only record legend once (from the first subplot)
        if not global_handles:
            global_handles = [h_preal, h_prpre, h_deal, h_depre]
            global_labels  = [
                "Prefill – Actual",
                "Prefill – Predicted",
                "Decode – Actual",
                "Decode – Predicted"
            ]

        # Error box
        prefill_err_pct = np.nanmean(np.abs(prefill_predicted - prefill_actual) / prefill_actual * 100)
        decode_err_pct  = np.nanmean(np.abs(decode_predicted - decode_actual) / decode_actual * 100)
        decode_err_diff = np.nanmean(np.abs(decode_predicted - decode_actual))

        textstr = (
            f"Prefill Error: {prefill_err_pct:.2f}%\n"
            f"Decode Error:  {decode_err_pct:.2f}% (+{decode_err_diff:.4f}s)"
        )

        ax.text(
            0.01, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8)
        )

        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xticks(np.arange(num_points))
    axes[-1].set_xticklabels([str(l) for l in layouts], rotation=45, fontsize=9)
    axes[-1].set_xlabel("Batch Layout (#Inf Reqs, #FT Samples)")

    fig.supylabel("Execution Time (s)", fontsize=12)

    # ------------------------------------------------------------
    # Global legend with 4 entries (prefill/decode + actual/pred)
    # ------------------------------------------------------------
    fig.legend(
        handles=global_handles,
        labels=global_labels,
        loc="upper center",
        ncol=2,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=200)

    out_pdf = "profiling.pdf"
    print(f"Saving figure to {out_pdf}")
    plt.savefig(out_pdf, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    plot_two_files(
        "batch_prediction_stats_5090.csv",
        "batch_prediction_stats_A100.csv",
        out_path="batch_prediction_stats.png"
    )