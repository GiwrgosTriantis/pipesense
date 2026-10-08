"""Shared helpers for the PipeSense batch pipeline.
"""

import json
import os
import time
from functools import wraps

# accumulated stage durations for the run report (appended by @log_time)
durations = []

GREEN = "\033[92m"
ENDC = "\033[0m"


def print_banner(msg, color=GREEN):
    """Print a colored banner line so stages are visible in the VM logs."""
    print(f"{color}{'=' * 70}{ENDC}")
    print(f"{color}{msg}{ENDC}")
    print(f"{color}{'=' * 70}{ENDC}")


def log_time(stage_name):
    """Decorator that times a stage and appends it to the global durations list."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            durations.append({"stage": stage_name, "duration_sec": round(elapsed, 2)})
            print_banner(f"{stage_name} finished in {elapsed:.1f}s")
            return result

        return wrapper

    return decorator


def get_work_dir():
    """Resolve the shared work directory used to pass state between stages."""
    work_dir = os.environ.get("WORK_DIR", "./output")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def get_data_dir():
    """Resolve the input data directory."""
    return os.environ.get("DATA_DIR", "./data")


def read_table(path):
    """Read a csv into a dataframe."""
    assert os.path.exists(path), f"input file not found: {path}"
    import pandas as pd

    return pd.read_csv(path)


def write_status(status, error=None):
    """Write the run status json that the VM monitor polls every 30 seconds."""
    payload = {"status": status, "durations": durations}
    if error is not None:
        payload["error"] = error
    with open(os.path.join(get_work_dir(), "status.json"), "w") as f:
        json.dump(payload, f, indent=2)
