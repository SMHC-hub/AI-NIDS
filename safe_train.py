import subprocess
import sys
import time
from pathlib import Path

import psutil

PROJECT_DIR = Path(__file__).resolve().parent

# Safety threshold below the 12 GiB budget
MEMORY_LIMIT_GIB = 11.0
CHECK_INTERVAL_SECONDS = 0.5


def process_tree_memory_gib(process):
    """Measure resident RAM used by a process and its children."""
    total_bytes = 0

    try:
        processes = [process] + process.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0.0

    for child in processes:
        try:
            total_bytes += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return total_bytes / (1024 ** 3)


def stop_process_tree(process):
    """Stop training and any child processes."""
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
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


def main():
    print("AI-NIDS Memory-Safe Training Launcher", flush=True)
    print(f"Memory warning/stop threshold: {MEMORY_LIMIT_GIB} GiB", flush=True)

    training = subprocess.Popen(
        [sys.executable, "-u", "train_model.py"],
        cwd=PROJECT_DIR,
    )

    process = psutil.Process(training.pid)
    peak_gib = 0.0

    try:
        while training.poll() is None:
            current_gib = process_tree_memory_gib(process)
            peak_gib = max(peak_gib, current_gib)

            print(
                f"\rTraining RAM: {current_gib:.2f} GiB "
                f"| Peak: {peak_gib:.2f} GiB",
                end="",
                flush=True,
            )

            if current_gib >= MEMORY_LIMIT_GIB:
                print("\nWARNING: Memory threshold reached!")
                stop_process_tree(process)
                print("Training stopped to protect system memory.")
                sys.exit(1)

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        stop_process_tree(process)
        sys.exit(130)

    print(f"\nPeak observed training RAM: {peak_gib:.2f} GiB")

    if training.returncode != 0:
        print(f"Training exited with code {training.returncode}")
        sys.exit(training.returncode)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()