import os
import json
import random
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
PROJECT_ID = "PROJECT ID"      # Default GCP project
INSTANCE_NAME_BASE = "gpu-worker"
REGION_FILTER = "europe"           # Restrict discovery to matching regions
MAX_RETRIES = -1                   # -1 means endless attempts
RETRY_DELAY = 120                  # Pause in seconds between waves
MAX_WORKERS = 6                    # Concurrent attempts

# GPU model -> machine type mapping
GPU_CONFIG = {
    "nvidia-tesla-t4": "n1-standard-4",
    "nvidia-l4": "g2-standard-4"
}

# Globals used for coordination across threads
stop_event = threading.Event()
print_lock = threading.Lock()

# Copy current environment and ensure gcloud sees the desired project
GCLOUD_ENV = os.environ.copy()
GCLOUD_ENV.setdefault("CLOUDSDK_CORE_PROJECT", PROJECT_ID)

def log(msg: str) -> None:
    """Thread-safe stdout logging."""
    with print_lock:
        print(msg)

def run_gcloud_json(cmd_list: Sequence[str]) -> list[dict]:
    """Execute gcloud and return parsed JSON, swallowing errors for discovery."""
    try:
        cmd = list(cmd_list) + ["--format=json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=GCLOUD_ENV)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError:
        return []

def warn_if_project_mismatch():
    """Warn when the active gcloud configuration points to another project."""
    cmd = ["gcloud", "config", "get-value", "project"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=GCLOUD_ENV)
        current = result.stdout.strip()
        if current in {"", "(unset)"}:
            log(f"⚠️ No active gcloud project. Forcing {PROJECT_ID} via CLOUDSDK_CORE_PROJECT. Consider running `gcloud config set project {PROJECT_ID}`.")
        elif current != PROJECT_ID:
            log(f"⚠️ Active gcloud project '{current}' differs. Overriding locally to {PROJECT_ID}. Run `gcloud config set project {PROJECT_ID}` to persist.")
    except FileNotFoundError as exc:
        log(f"⚠️ gcloud executable not found in PATH: {exc}")
    except subprocess.CalledProcessError as exc:
        log(f"⚠️ Unable to read gcloud project: {exc.stderr.strip() if exc.stderr else exc}")

def get_zones_for_gpu(gpu_type):
    """Return zones supporting the GPU while stripping long resource URLs."""
    cmd = [
        "gcloud", "compute", "accelerator-types", "list",
        f"--project={PROJECT_ID}",
        f"--filter=name={gpu_type} AND zone:({REGION_FILTER})"
    ]
    data = run_gcloud_json(cmd)

    zones = [item["zone"].split("/")[-1] for item in data if "zone" in item]
    return sorted(set(zones))

def create_vm(zone, gpu_type):
    """Attempt a VM creation, aborting if another worker already succeeded."""
    if stop_event.is_set():
        return False

    machine_type = GPU_CONFIG[gpu_type]
    instance_name = f"{INSTANCE_NAME_BASE}-{gpu_type.split('-')[-1]}-{zone}"

    log(f"[Start] Attempting {gpu_type} in {zone}...")

    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--accelerator=type={gpu_type},count=1",
        "--maintenance-policy=TERMINATE",
        "--image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=200GB",
        "--quiet"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=GCLOUD_ENV)

    if stop_event.is_set():
        return False

    if result.returncode == 0:
        stop_event.set()
        log(f"\n[SUCCESS] GPU {gpu_type} available in {zone}.")
        log(f"[SSH] gcloud compute ssh {instance_name} --zone={zone}")
        return True
    else:
        err = result.stderr
        if "resources" in err or "exhausted" in err or "not available" in err:
            log(f"[Fail] {zone} ({gpu_type}): capacity exhausted.")
        elif "quota" in err.lower():
            log(f"[Fail] {zone} ({gpu_type}): quota error, review the console.")
        elif "invalid" in err.lower() or "not found" in err.lower():
            log(f"[Error] {zone}: {err.strip().splitlines()[-1]}")
        else:
            log(f"[Fail] {zone} ({gpu_type}): {err.strip() or 'Unexpected error.'}")
        return False

def main():
    attempt_count = 0

    warn_if_project_mismatch()

    log("[Init] Discovering eligible zones...")
    tasks = []
    for gpu_type in GPU_CONFIG:
        zones = get_zones_for_gpu(gpu_type)
        log(f"   {gpu_type}: {len(zones)} zone(s) detected.")
        tasks.extend((z, gpu_type) for z in zones)

    if not tasks:
        log("[Error] No zones found. Verify PROJECT_ID or IAM permissions.")
        sys.exit(1)

    log(f"[Init] Total combinations to test: {len(tasks)}")

    while not stop_event.is_set():
        attempt_count += 1
        log(f"\n[Wave {attempt_count}] Running with {MAX_WORKERS} workers...")

        random.shuffle(tasks)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for zone, gpu in tasks:
                if stop_event.is_set():
                    break
                futures.append(executor.submit(create_vm, zone, gpu))

            for f in as_completed(futures):
                try:
                    f.result()
                except FileNotFoundError as exc:
                    log(f"[Error] gcloud not found (PATH issue?): {exc}")
                    stop_event.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as exc:
                    log(f"[Error] Worker crashed: {exc}")

        if stop_event.is_set():
            break

        if MAX_RETRIES != -1 and attempt_count >= MAX_RETRIES:
            log("\n[Stop] No GPU found before reaching the retry limit.")
            break

        if not stop_event.is_set():
            log(f"\n[Sleep] Waiting {RETRY_DELAY}s before the next wave...")
            time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    main()
