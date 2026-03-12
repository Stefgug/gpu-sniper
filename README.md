# VM Sniper

Simple Python script that keeps trying to create a GPU VM on Google Cloud until capacity becomes available.

## What it does

- Looks up zones that support your target GPU types.
- Tries VM creation in parallel across zones.
- Repeats in waves until one VM is created (or retry limit is reached).

## Prerequisites

- Python 3.10+
- Google Cloud SDK (`gcloud`) installed and in your PATH
- A GCP project with Compute Engine API enabled
- IAM permissions to create Compute Engine instances
- GPU quota in your project (or enough quota for your requested resources)

## 1) Configure the script

Open [sniping.py](sniping.py) and edit the values under the `# --- CONFIGURATION ---` section:

- `PROJECT_ID`: your GCP project ID
- `INSTANCE_NAME_BASE`: base name for created instances
- `REGION_FILTER`: region substring used when discovering zones (example: `europe`)
- `MAX_RETRIES`: number of waves (`-1` for infinite)
- `RETRY_DELAY`: seconds to wait between waves
- `MAX_WORKERS`: parallel VM creation attempts per wave
- `IMAGE_FAMILY`: image family for the VM boot disk
- `IMAGE_PROJECT`: image project for that image family
- `GPU_CONFIG`: mapping of GPU type to machine type

## 2) Authenticate gcloud

If needed:

```bash
gcloud auth login
gcloud auth application-default login
```

Optionally set your default project globally:

```bash
gcloud config set project YOUR_PROJECT_ID
```

The script also forces the configured project via environment variable, and warns if your active gcloud project differs.

## 3) Run

From the repository root:

```bash
python3 sniping.py
```

## 4) Success output

When capacity is found, the script prints:

- the GPU type and zone that succeeded
- an SSH command you can run to connect to the new instance

Example:

```text
[SUCCESS] GPU nvidia-l4 available in europe-west4-b.
[SSH] gcloud compute ssh gpu-worker-l4-europe-west4-b --zone=europe-west4-b
```

## Stopping the script

Press `Ctrl+C` to stop manually.
