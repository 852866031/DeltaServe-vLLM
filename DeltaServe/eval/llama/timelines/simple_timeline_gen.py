import csv
import os

def generate_request_pattern(rps: int, duration: int, prompt_length: int, max_new_tokens: int, output_file: str):
    """
    Generate a CSV file simulating evenly distributed request traces.

    Each second contains `rps` requests spaced evenly across the second,
    e.g., for rps=4:
        timestamps = 0.00, 0.25, 0.50, 0.75 for second 0,
                     1.00, 1.25, 1.50, 1.75 for second 1, etc.

    Args:
        rps (int): requests per second
        duration (int): total duration in seconds
        prompt_length (int): prompt token length
        max_new_tokens (int): maximum new tokens
        output_file (str): path to output CSV file
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "prompt_length", "max_new_tokens", "second", "index_in_second"])

        interval = 1.0 / rps  # evenly spaced interval within one second
        total_requests = 0

        for second in range(duration):
            for i in range(rps):
                timestamp = round(second + i * interval, 6)
                writer.writerow([timestamp, prompt_length, max_new_tokens, second, i])
                total_requests += 1

    print(f"✅ Generated {output_file} with {total_requests} total requests "
          f"({rps}/s for {duration}s, evenly distributed).")


if __name__ == "__main__":
    # Example usage
    requests_per_second = 12
    duration_seconds = 8
    prompt_length = 40
    max_new_tokens = 42
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    output_filename = f"{current_directory}/timeline_gen.csv"
    generate_request_pattern(
        requests_per_second,
        duration_seconds,
        prompt_length,
        max_new_tokens,
        output_filename
    )