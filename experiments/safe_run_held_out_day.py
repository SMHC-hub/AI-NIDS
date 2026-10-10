"""Memory-Safe Experiment Launcher for AI-NIDS Held-Out-Day Experiment.

Follows the exact pattern from safe_train.py:
- Spawns experiments/held_out_day.py with unbuffered output (-u)
- Monitors resident memory (RSS) of the process tree every 0.5 seconds
- Enforces an 11.0 GiB safety ceiling
- Safely terminates all child processes if the threshold is reached
- Reports peak observed RAM upon completion
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

PROJECT_DIR = Path(__file__).resolve().parents[1]

MEMORY_LIMIT_GIB = 11.0
CHECK_INTERVAL_SECONDS = 0.5


def process_tree_memory_gib(process: psutil.Process) -> float:
    """Measure resident RAM used by a process and all its children."""
    total_bytes = 0

    try:
        processes = [process] + process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0

    for child in processes:
        try:
            total_bytes += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return total_bytes / (1024 ** 3)


def stop_process_tree(process: psutil.Process) -> None:
    """Stop the experiment and all child worker processes."""
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []

    for child in children:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    try:
        process.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    psutil.wait_procs(children + [process], timeout=5)


def main() -> None:
    print("=" * 60, flush=True)
    print("AI-NIDS Held-Out-Day Memory-Safe Experiment Runner", flush=True)
    print(f"Safety memory limit: {MEMORY_LIMIT_GIB:.1f} GiB", flush=True)
    print("Target: experiments/held_out_day.py", flush=True)
    print("=" * 60, flush=True)

    cmd = [sys.executable, "-u", str(PROJECT_DIR / "experiments" / "held_out_day.py")]

    process_pop = subprocess.Popen(
        cmd,
        cwd=PROJECT_DIR,
    )

    proc = psutil.Process(process_pop.pid)
    peak_gib = 0.0

    try:
        while process_pop.poll() is None:
            current_gib = process_tree_memory_gib(proc)
            peak_gib = max(peak_gib, current_gib)

            print(
                f"\r[RAM Monitor] Current: {current_gib:.2f} GiB | Peak: {peak_gib:.2f} GiB / {MEMORY_LIMIT_GIB:.1f} GiB",
                end="",
                flush=True,
            )

            if current_gib >= MEMORY_LIMIT_GIB:
                print(f"\nCRITICAL: Memory limit exceeded ({current_gib:.2f} GiB >= {MEMORY_LIMIT_GIB:.1f} GiB)!")
                stop_process_tree(proc)
                print("Experiment safely halted to prevent system exhaustion.")
                sys.exit(1)

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        stop_process_tree(proc)
        sys.exit(130)

    print(f"\n\nPeak RAM observed during entire experiment: {peak_gib:.2f} GiB")

    if process_pop.returncode != 0:
        print(f"Experiment exited with error code {process_pop.returncode}")
        sys.exit(process_pop.returncode)

    print("Experiment completed successfully within memory budget.")


if __name__ == "__main__":
    main()
