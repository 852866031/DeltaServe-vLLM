import asyncio
from datetime import datetime
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature,
    NVML_TEMPERATURE_GPU,
)
from typing import Optional, List


async def monitor_gpu_usage(
    log_path: str = "gpu_usage_log.csv",
    interval_s: float = 0.2,
    aggregation_period_s: float = 0.5,     # NEW ✓ group samples into windows
    stop_event: Optional[asyncio.Event] = None,
):
    """
    High-resolution GPU monitor:
      • Samples GPU every interval_s
      • Aggregates samples into windows of aggregation_period_s
      • Writes only the peak values (per GPU) for each window
      • Column name = timestamp (string)
    """

    print(f"[monitor] Starting GPU monitor → aggregation window = {aggregation_period_s}s")

    # CSV header
    buffer: List[str] = []
    buffer.append(
        "timestamp,gpu_index,gpu_util,"
        "memory_used_mb,memory_total_mb,power_w,temperature_c\n"
    )

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        print(f"[monitor] {device_count} GPU(s) detected → writing to {log_path}")

        # Track window boundaries
        t0 = datetime.now().timestamp()
        window_index = 0  # incremented each window

        # Accumulators: gpu_id → dict of peak metrics
        accumulators = {}

        while True:
            if stop_event is not None and stop_event.is_set():
                break

            now = datetime.now()
            now_ts = now.timestamp()

            # Determine current window index
            current_window = int((now_ts - t0) // aggregation_period_s)

            # If window changed → flush previous window
            if current_window != window_index and accumulators:
                # Window closing → write peaks
                timestamp_str = datetime.fromtimestamp(
                    t0 + window_index * aggregation_period_s
                ).strftime("%Y-%m-%d %H:%M:%S")

                for gpu_id, peak in accumulators.items():
                    buffer.append(
                        f"{timestamp_str},{gpu_id},{peak['util']},"
                        f"{peak['mem_used']:.1f},{peak['mem_total']:.1f},"
                        f"{peak['power']:.1f},{peak['temp']}\n"
                    )

                # Reset for next window
                window_index = current_window
                accumulators = {}

            # Sample GPUs and record peaks
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle).gpu
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

                acc = accumulators.setdefault(i, {
                    "util": 0,
                    "mem_used": 0,
                    "mem_total": mem.total / 1024**2,
                    "power": 0,
                    "temp": 0,
                })

                acc["util"] = max(acc["util"], util)
                acc["mem_used"] = max(acc["mem_used"], mem.used / 1024**2)
                acc["power"] = max(acc["power"], power)
                acc["temp"] = max(acc["temp"], temp)

            # Sleep while remaining cancellable
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                print("[monitor] Cancelled mid-sleep.")
                break

    except KeyboardInterrupt:
        print("[monitor] KeyboardInterrupt → stopping.")
    finally:
        try:
            nvmlShutdown()
        except:
            pass

        # Flush last partial window if any
        try:
            if accumulators:
                timestamp_str = datetime.fromtimestamp(
                    t0 + window_index * aggregation_period_s
                ).strftime("%Y-%m-%d %H:%M:%S")

                for gpu_id, peak in accumulators.items():
                    buffer.append(
                        f"{timestamp_str},{gpu_id},{peak['util']},"
                        f"{peak['mem_used']:.1f},{peak['mem_total']:.1f},"
                        f"{peak['power']:.1f},{peak['temp']}\n"
                    )
        except:
            pass

        # Write log file
        try:
            with open(log_path, "w") as f:
                f.writelines(buffer)

            print(f"[monitor] Stopped — wrote {len(buffer)-1} rows → {log_path}")
        except Exception as e:
            print(f"[monitor] Failed to write log: {e}")
