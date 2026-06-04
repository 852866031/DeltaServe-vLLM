import csv
import os
import subprocess
from itertools import cycle

def generate_peak_valley_timeline(
    peak_duration: float,
    valley_duration: float,
    prompt_length: int,
    max_new_tokens: int,
    total_cycles: int,
    output_file: str,
    peak_rps,
    valley_rps,
    initial_idle: float = 1.0,
):
    """
    Generate a CSV timeline simulating alternating 'peak' and 'valley' request periods,
    with cycling peak/valley RPS values.

    Example:
        peak_rps=[4,5] → peak 1 uses 4 rps, peak 2 uses 5, then 4, 5, ...
        valley_rps=[1,2] → valley 1 uses 1 rps, valley 2 uses 2, then 1, 2, ...
    """

    # --- Convert integers to 1-element list so both behave consistently ---
    if isinstance(peak_rps, int):
        peak_rps = [peak_rps]
    if isinstance(valley_rps, int):
        valley_rps = [valley_rps]

    peak_cycle = cycle(peak_rps)
    valley_cycle = cycle(valley_rps)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "prompt_length", "max_new_tokens", "second", "index_in_second"])

        time = initial_idle
        total_reqs = 0

        for cycle_idx in range(total_cycles):
            # === Valley Phase ===
            current_valley_rps = next(valley_cycle)
            valley_count = int(valley_duration * current_valley_rps)

            for i in range(valley_count):
                timestamp = time + i / max(1, current_valley_rps)
                second = int(timestamp)
                writer.writerow([round(timestamp, 6), prompt_length, max_new_tokens, second, i % max(1, current_valley_rps)])
                total_reqs += 1

            time += valley_duration

             # === Peak Phase ===
            current_peak_rps = next(peak_cycle)
            peak_count = int(peak_duration * current_peak_rps)

            for i in range(peak_count):
                timestamp = time + i / current_peak_rps
                second = int(timestamp)
                writer.writerow([round(timestamp, 6), prompt_length, max_new_tokens, second, i % current_peak_rps])
                total_reqs += 1

            time += peak_duration

    print(f"✅ Generated {output_file} with {total_reqs} requests.")

    # === Automatically call plot_timeline.py ===
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_script = os.path.join(current_dir, "plot_timeline.py")
        if not os.path.exists(plot_script):
            print(f"⚠️ plot_timeline.py not found at {plot_script}, skipping plot.")
            return
        print(f"📈 Calling plot_timeline.py...")
        subprocess.run(["python", plot_script, output_file], check=True)
        print(f"✅ Plot generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to plot timeline: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error while plotting: {e}")


if __name__ == "__main__":
    peak_duration = 2.0
    valley_duration = 4.0
    prompt_length = 80
    max_new_tokens = 80
    total_cycles = 8
    peak_rps = [14]
    valley_rps = [0]

    initial_idle = 1.0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "timeline_gen.csv")

    generate_peak_valley_timeline(
        peak_duration,
        valley_duration,
        prompt_length,
        max_new_tokens,
        total_cycles,
        output_path,
        peak_rps=peak_rps,
        valley_rps=valley_rps,
        initial_idle=initial_idle,
    )