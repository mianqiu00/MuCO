# MuCO Backend Deployment

This document is the backend-only deployment handoff for the MuCO inference API.

## What Is Included

- `Dockerfile.api`: CUDA 11.8 runtime image that creates the Conda environment from `environment.yml`.
- `docker-compose.api.yml`: backend-only service exposing NVIDIA GPUs to the container and selecting GPU 0 via `NVIDIA_VISIBLE_DEVICES=0` and `MUCO_DEVICE=0`.
- `muco_api.py`: resident FastAPI service that loads MuCO models once at startup and runs jobs through a single-GPU queue.
- `backend/config/settings.py`: server-side configuration for job storage, logs, limits, downloads, Redis URL, and model path.
- `params/link.md`: download link for pretrained checkpoints. The checkpoint files themselves are not committed.

## Server Requirements

- Linux server with NVIDIA GPU.
- NVIDIA driver compatible with CUDA 11.8.
- Docker with NVIDIA Container Toolkit installed.
- Model weights present under `params/`:
  - `params/foldflow.pth`
  - `params/flowpacker.pth`

Download the pretrained checkpoints from the Google Drive link in `params/link.md`, then place the `.pth` files under `params/` before starting the API container. The `params/` directory is mounted read-only into the container and ignored by Git except for the link file.

Check GPU availability on the server:

```bash
nvidia-smi
```

## Build Backend Image

From the repository root:

```bash
docker compose -f docker-compose.api.yml build
```

If the server does not have Docker Compose v2, build with plain Docker:

```bash
docker build -f Dockerfile.api -t muco-api:latest .
```

The built image is tagged as:

```text
muco-api:latest
```

Python, PyTorch, FastAPI, OpenMM, PyMOL and model dependencies are centralized in `environment.yml`. If a dependency changes, update `environment.yml` first and rebuild the image instead of adding one-off install commands to the Dockerfile.

## Run Backend On GPU 0

```bash
docker compose -f docker-compose.api.yml up -d
```

Equivalent plain Docker command:

```bash
docker run -d \
  --name muco-api \
  --gpus '"device=0"' \
  --restart unless-stopped \
  -p 8000:8000 \
  -e MUCO_DEVICE=0 \
  -e JOB_ROOT=/data/muco/jobs \
  -e LOG_ROOT=/data/muco/logs \
  -e TORCH_HOME=/opt/torch-cache \
  -v "$PWD/params:/srv/muco/params:ro" \
  -v "$PWD/runs/jobs:/data/muco/jobs" \
  -v "$PWD/runs/logs:/data/muco/logs" \
  -v "$PWD/torch-cache:/opt/torch-cache" \
  muco-api:latest
```

The API listens on:

```text
http://SERVER_IP:8000
```

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/resources
```

Expected `/ready` response after model loading:

```json
{
  "ready": true,
  "cuda_available": true,
  "device": 0
}
```

## Submit A Test Job

This test uses GPU 0 and runs all three MuCO stages, including OpenMM relaxation.

```bash
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'content-type: application/json' \
  -d '{
    "sequence": "ACDEFGHIK",
    "K": 1,
    "M": 1,
    "backbone_steps": 10,
    "sidechain_steps": 3,
    "make_zip": false
  }'
```

The response contains a `job_id`:

```json
{
  "job_id": "20260514Txxxxxx_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "queued"
}
```

Poll the job:

```bash
curl http://127.0.0.1:8000/jobs/JOB_ID
```

Get summary after completion:

```bash
curl http://127.0.0.1:8000/jobs/JOB_ID/summary
```

Get logs:

```bash
curl -o muco.log http://127.0.0.1:8000/jobs/JOB_ID/logs
```

## Server-Managed Job Storage

The API does not accept filesystem paths from clients. Each job is stored under server-controlled directories:

```text
JOB_ROOT/{job_id}
LOG_ROOT/{job_id}
```

Default Docker paths are:

```text
/data/muco/jobs/{job_id}
/data/muco/logs/{job_id}
```

These map to the host repository directory:

```text
./runs/jobs/{job_id}
./runs/logs/{job_id}
```

The command-line inference wrapper supports the same split between outputs and logs:

```bash
python muco_infer.py input.json \
  --output /srv/muco/runs/cli/job-001 \
  --log_dir /srv/muco/runs/cli_logs/job-001 \
  --K 1 \
  --M 1
```

This writes generated files under `--output` and runtime logs to `--log_dir/muco.log`.

## Resource Usage

Current process and GPU memory snapshot:

```bash
curl http://127.0.0.1:8000/resources
```

Per-job status responses intentionally omit server internals such as filesystem paths and GPU details.

For live GPU monitoring on the server:

```bash
nvidia-smi -i 0 -l 1
```

For Docker CPU and memory monitoring:

```bash
docker stats muco-api
```

View container logs:

```bash
docker logs -f muco-api
```

## API Endpoints

- `GET /health`: process health.
- `GET /ready`: model loading and CUDA readiness.
- `GET /resources`: CPU/GPU memory snapshot.
- `POST /jobs`: submit inference job.
- `GET /jobs/{job_id}`: job status and progress.
- `GET /jobs/{job_id}/summary`: final generated structure summary.
- `GET /jobs/{job_id}/download`: successful relaxed PDB zip.
- `GET /jobs/{job_id}/files?path=pdb_K_M|png_K_M`: fetch an output file by token.
- `GET /jobs/{job_id}/logs`: fetch job log.

## Runtime Notes

- The service loads both checkpoints once at startup.
- Jobs are executed by a single-worker queue to avoid multiple tasks fighting for GPU 0.
- Side-chain packing currently requires CUDA because the existing loader calls `.cuda()`.
- For quick smoke tests, use small `backbone_steps`, small `sidechain_steps`, `K=1`, and `M=1`. Stage 3 relaxation is always executed.
