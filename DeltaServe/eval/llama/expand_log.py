import pandas as pd
import numpy as np


# ============================================================
# 1. Expand FT 1-minute log → N minutes
# ============================================================
def expand_ft_log_one_minute_to_20min(input_csv, output_csv, minutes=20):
    """
    Expand a 1-minute backward FT log to N minutes by continuing its trend.
    """

    # ------------------------------------------------------------
    # Load original FT log
    # ------------------------------------------------------------
    df = pd.read_csv(input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])   # KEEP as pandas timestamps

    # Extract timestamps + tokens
    t = df["timestamp"]
    tokens = df["total_processed_tokens"]

    # ---- compute dt safely (pandas Timedelta) ----
    time_deltas = t.diff().dt.total_seconds().iloc[1:]
    avg_dt = time_deltas.mean()
    if not np.isfinite(avg_dt):
        avg_dt = 1.0
    avg_dt = max(avg_dt, 0.1)

    # ---- compute token increments ----
    token_deltas = tokens.diff().iloc[1:]
    avg_tokens = token_deltas.mean()
    if not np.isfinite(avg_tokens):
        avg_tokens = 1
    avg_tokens = max(avg_tokens, 1)

    # ------------------------------------------------------------
    # Extend data
    # ------------------------------------------------------------
    t0 = t.iloc[0]
    current_t = t.iloc[-1]
    current_tokens = tokens.iloc[-1]

    total_seconds = minutes * 60

    rows = []

    while (current_t - t0).total_seconds() < total_seconds:
        # increment timestamp
        current_t = current_t + pd.Timedelta(seconds=float(avg_dt))
        # increment tokens
        current_tokens = current_tokens + avg_tokens

        rows.append({
            "timestamp": current_t.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": "",
            "batch_idx": "",
            "batch_tokens": "",
            "batch_loss": "",
            "total_processed_tokens": int(current_tokens),
        })

    # ------------------------------------------------------------
    # Combine original + generated
    # ------------------------------------------------------------
    df_base = df.copy()
    df_base["epoch"] = ""
    df_base["batch_idx"] = ""
    df_base["batch_tokens"] = 256
    df_base["batch_loss"] = ""

    df_base = df_base[
        ["timestamp", "epoch", "batch_idx", "batch_tokens", "batch_loss", "total_processed_tokens"]
    ]

    df_new = pd.DataFrame(rows)

    df_final = pd.concat([df_base, df_new], ignore_index=True)

    df_final.to_csv(output_csv, index=False)
    print(f"✅ Generated extended FT log ({minutes} min) → {output_csv}")


# ============================================================
# 2. Expand GPU log by repeating its pattern
# ============================================================
def expand_gpu_log_pattern(input_csv, output_csv, minutes=20):
    """
    Extend a 1-minute GPU usage log to N minutes by repeating the pattern.
    """

    df = pd.read_csv(input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute dt from timestamps
    time_deltas = df["timestamp"].diff().dt.total_seconds().iloc[1:]
    avg_dt = time_deltas.mean()
    if not np.isfinite(avg_dt):
        avg_dt = 1.0
    avg_dt = max(avg_dt, 1.0)

    total_seconds = minutes * 60

    # Extract rows as pattern
    pattern = df.to_dict("records")
    pattern_len = len(pattern)

    rows = []
    current_t = df["timestamp"].iloc[0]
    elapsed = 0.0
    idx = 0

    while elapsed < total_seconds:
        base = pattern[idx]

        rows.append({
            "timestamp": current_t.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_index": base["gpu_index"],
            "gpu_util": base["gpu_util"],
            "memory_used_mb": base["memory_used_mb"],
            "memory_total_mb": base["memory_total_mb"],
            "power_w": base["power_w"],
            "temperature_c": base["temperature_c"],
        })

        # advance time
        current_t = current_t + pd.Timedelta(seconds=float(avg_dt))
        elapsed += avg_dt

        # cycle pattern
        idx = (idx + 1) % pattern_len

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    print(f"✅ Extended GPU usage log ({minutes} min) → {output_csv}")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    ft_log_path = "results/bwd_log_0.csv"
    ft_log_out_path = "results/bwd_log_0_extended_20min.csv"

    gpu_log_path = "results/gpu_usage_ft.csv"
    gpu_log_out_path = "results/gpu_usage_0_extended_20min.csv"

    expand_ft_log_one_minute_to_20min(
        ft_log_path,
        ft_log_out_path,
        minutes=20
    )

    expand_gpu_log_pattern(
        gpu_log_path,
        gpu_log_out_path,
        minutes=20
    )