import contextlib
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from backend.config.settings import settings
from muco_infer import _relax_pdb, _write_success_zip
from runner.backbone_sampler import BackboneSampler
from runner.sidechain_sampler import SidechainSampler
from utils import seed_everything


ROOT = Path(__file__).resolve().parent
AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWYX")
PYMOL_RENDER_SCRIPT = ROOT / "app" / "pymol" / "render_pdb.py"


class MucoJobRequest(BaseModel):
    sequence: str = Field(..., min_length=2, max_length=30)
    K: int = Field(1, ge=1, le=settings.max_k)
    M: int = Field(1, ge=1, le=settings.max_m)
    seed: int = 42
    backbone_steps: int = Field(100, ge=2)
    sidechain_steps: int = Field(10, ge=1)
    sidechain_coeff: float = 5.0
    noise_scale: float = 1.0
    min_t: float = 0.01
    relax_platform: str = "CPU"
    relax_log: bool = False
    make_zip: bool = True

    class Config:
        extra = "forbid"


class MucoJobResponse(BaseModel):
    job_id: str
    status: str


class MucoService:
    def __init__(self):
        self.device = settings.device
        self.seed = settings.seed
        self.backbone_steps = settings.backbone_steps
        self.sidechain_steps = settings.sidechain_steps
        self.sidechain_coeff = settings.sidechain_coeff
        self.noise_scale = settings.noise_scale
        self.min_t = settings.min_t
        self.use_gt_masks = settings.use_gt_masks
        self.lock = threading.Lock()
        self.backbone_sampler = None
        self.sidechain_sampler = None

    def load(self):
        seed_everything(self.seed)
        self.backbone_sampler = BackboneSampler(
            conf=self._backbone_config(
                backbone_steps=self.backbone_steps,
                noise_scale=self.noise_scale,
                min_t=self.min_t,
            ),
            ckpt_epoch=100,
            output_dir=str(settings.job_root / "_warm" / "stage1_backbone"),
            device_id=self.device,
            is_sample=True,
            batch_size=1,
        )

        self.sidechain_sampler = SidechainSampler(
            self._sidechain_config(
                seed=self.seed,
                sidechain_steps=self.sidechain_steps,
                sidechain_coeff=self.sidechain_coeff,
                work_dir=settings.job_root / "_warm",
            ),
            use_gt_masks=self.use_gt_masks,
        )
        self.sidechain_sampler._load_model_for_inference()
        self._sync_cuda()

    def _backbone_config(self, backbone_steps, noise_scale, min_t):
        conf = OmegaConf.load(ROOT / "config" / "backbone.yaml")
        conf.experiment.use_gpu = torch.cuda.is_available() and self.device >= 0
        conf.experiment.batch_size = 1
        conf.experiment.eval_batch_size = 1
        conf.experiment.noise_scale = noise_scale
        conf.experiment.full_ckpt_dir = str(settings.model_path / "foldflow.pth")
        conf.data.num_t = backbone_steps
        conf.data.min_t = min_t
        conf.data.cache_full_dataset = False
        conf.data.cache_dataset_in_memory = False
        return conf

    def _sidechain_config(self, seed, sidechain_steps, sidechain_coeff, work_dir):
        ckpt_path = settings.model_path / "flowpacker.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError("Missing FlowPacker checkpoint: %s" % ckpt_path)
        if not torch.cuda.is_available():
            raise RuntimeError("FlowPacker sampler requires CUDA because the existing model loader calls .cuda().")

        ckpt_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        config = ckpt_dict["config"]
        config.ckpt = str(ckpt_path)
        config.seed = seed
        config.direct_inference = True
        if getattr(config, "mode", None) == "flow":
            config.mode = getattr(config.train, "loss_type", "vf")
        config.conf_ckpt = getattr(config, "conf_ckpt", None)
        if not hasattr(config, "sample"):
            config.sample = ConfigDict()
        config.sample.n_samples = 1
        config.sample.batch_size = 1
        config.sample.num_steps = sidechain_steps
        config.sample.coeff = sidechain_coeff
        config.data.test_path = str(Path(work_dir) / "sidechain_input")
        return config

    @staticmethod
    def _validate_sequence(sequence):
        sequence = sequence.strip().upper()
        bad = sorted(set(sequence) - AA_LETTERS)
        if not sequence:
            raise ValueError("Input sequence is empty.")
        if bad:
            raise ValueError("Unsupported residue letters: %s" % "".join(bad))
        if len(sequence) < 2 or len(sequence) > 30:
            raise ValueError("Input sequence length must be 2-30 residues.")
        return sequence

    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _copy_for_sidechain(backbone_pdb, sidechain_input_dir):
        sidechain_input_dir.mkdir(parents=True, exist_ok=True)
        target = sidechain_input_dir / Path(backbone_pdb).name
        shutil.copyfile(str(backbone_pdb), str(target))
        return target

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(str(tmp), "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(str(tmp), str(path))

    @staticmethod
    def _append_log(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "a") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

    @staticmethod
    def _render_pdb(pdb_path, png_path):
        png_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(PYMOL_RENDER_SCRIPT), str(pdb_path), str(png_path)],
            cwd=str(ROOT),
            check=True,
        )
        return str(png_path)

    def generate(self, request, job_dir, progress_callback=None):
        if self.backbone_sampler is None or self.sidechain_sampler is None:
            raise RuntimeError("MuCO service has not loaded models yet.")

        sequence = self._validate_sequence(request.sequence)
        work_dir = Path(job_dir).resolve()
        backbone_dir = work_dir / "stage1_backbone"
        sidechain_dir = work_dir / "stage2_sidechain"
        relaxed_dir = work_dir / "stage3_relaxed"
        sidechain_input_dir = work_dir / "sidechain_input"
        for path in [backbone_dir, sidechain_dir, relaxed_dir, sidechain_input_dir]:
            path.mkdir(parents=True, exist_ok=True)

        with self.lock:
            self.backbone_sampler._data_conf.num_t = int(request.backbone_steps)
            self.backbone_sampler._data_conf.min_t = float(request.min_t)
            self.backbone_sampler._exp_conf.noise_scale = float(request.noise_scale)
            self.sidechain_sampler.config.sample.num_steps = int(request.sidechain_steps)
            self.sidechain_sampler.config.sample.coeff = float(request.sidechain_coeff)
            if hasattr(self.sidechain_sampler, "model"):
                self.sidechain_sampler.model.stepsize = int(request.sidechain_steps)
                self.sidechain_sampler.model.coeff = float(request.sidechain_coeff)

            seed_everything(int(request.seed))
            backbone_jobs = [
                {"id": "peptide", "sequence": sequence, "k": k_idx, "run_name": "peptide_k%d" % k_idx}
                for k_idx in range(1, int(request.K) + 1)
            ]
            total_sidechains = len(backbone_jobs) * int(request.M)
            if progress_callback is not None:
                progress_callback("stage1", 0, len(backbone_jobs), 0, total_sidechains, 0, total_sidechains)

            self._sync_cuda()
            with torch.inference_mode():
                stage1_start = time.perf_counter()
                backbone_paths = self.backbone_sampler.sample_sequences_to_pdb(
                    [job["sequence"] for job in backbone_jobs],
                    [job["run_name"] for job in backbone_jobs],
                    str(backbone_dir),
                    progress_callback=(
                        lambda step, total: progress_callback(
                            "stage1", step, total, 0, total_sidechains, 0, total_sidechains
                        )
                        if progress_callback is not None else None
                    ),
                )
            self._sync_cuda()
            stage1_batch_seconds = time.perf_counter() - stage1_start
            stage1_seconds_each = stage1_batch_seconds / max(len(backbone_jobs), 1)

            results = []
            stage2_done = 0
            stage3_done = 0
            for job, backbone_pdb in zip(backbone_jobs, backbone_paths):
                sidechain_input = self._copy_for_sidechain(backbone_pdb, sidechain_input_dir)
                for m_idx in range(1, int(request.M) + 1):
                    packed_name = "peptide_k%d_m%d.pdb" % (job["k"], m_idx)
                    packed_pdb = sidechain_dir / packed_name
                    relaxed_pdb = relaxed_dir / packed_name
                    self._sync_cuda()
                    stage2_start = time.perf_counter()
                    with torch.inference_mode():
                        self.sidechain_sampler.sample_pdb_to_pdb(
                            sidechain_input,
                            packed_pdb,
                            progress_callback=None,
                        )
                    self._sync_cuda()
                    stage2_seconds = time.perf_counter() - stage2_start
                    stage2_done += 1
                    if progress_callback is not None:
                        progress_callback("stage2", len(backbone_jobs), len(backbone_jobs), stage2_done, total_sidechains, stage3_done, total_sidechains)

                    relax_info = None
                    stage3_seconds = 0.0
                    stage3_start = time.perf_counter()
                    relax_info = _relax_pdb(packed_pdb, relaxed_pdb, request.relax_platform, log=request.relax_log)
                    stage3_seconds = time.perf_counter() - stage3_start
                    stage3_done += 1
                    if progress_callback is not None:
                        progress_callback("stage3", len(backbone_jobs), len(backbone_jobs), stage2_done, total_sidechains, stage3_done, total_sidechains)

                    results.append({
                        "id": job["id"],
                        "sequence": sequence,
                        "length": len(sequence),
                        "k": job["k"],
                        "m": m_idx,
                        "stage1_seconds": stage1_seconds_each,
                        "stage1_batch_seconds": stage1_batch_seconds,
                        "stage1_batch_size": len(backbone_jobs),
                        "stage2_seconds": stage2_seconds,
                        "stage3_seconds": stage3_seconds,
                        "total_seconds": stage1_seconds_each + stage2_seconds + stage3_seconds,
                        "backbone_pdb": str(backbone_pdb),
                        "sidechain_pdb": str(packed_pdb),
                        "relaxed_pdb": str(relaxed_pdb),
                        "relax": relax_info,
                    })

            summary_path = work_dir / "summary.json"
            render_dir = work_dir / "renders"
            for row in results:
                pdb_for_render = row.get("relaxed_pdb") or row.get("sidechain_pdb")
                if not pdb_for_render:
                    continue
                png_path = render_dir / (Path(pdb_for_render).stem + ".png")
                try:
                    row["render_png"] = self._render_pdb(pdb_for_render, png_path)
                except Exception as exc:
                    row["render_error"] = str(exc)
            self._write_json(summary_path, results)
            if request.make_zip:
                zip_args = SimpleNamespace(make_zip=True, work_dir=work_dir)
                zip_path = _write_success_zip(zip_args, results)
                if zip_path:
                    for row in results:
                        row["success_zip"] = zip_path
                    self._write_json(summary_path, results)
            if progress_callback is not None:
                progress_callback("done", len(backbone_jobs), len(backbone_jobs), stage2_done, total_sidechains, stage3_done, total_sidechains)
            return str(summary_path)


service = MucoService()
app = FastAPI(title="MuCO API", version="0.1.0")
executor = ThreadPoolExecutor(max_workers=1)
jobs: Dict[str, Dict[str, object]] = {}
jobs_lock = threading.Lock()


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_, exc):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


def _job_dir(job_id):
    return settings.job_root / job_id


def _log_dir(job_id):
    return settings.log_root / job_id


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _resource_snapshot():
    snapshot = {
        "cpu_max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "cuda_available": torch.cuda.is_available(),
        "device": service.device,
    }
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % service.device)
        snapshot.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
            "gpu_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2),
            "gpu_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024 / 1024, 2),
        })
    return snapshot


def _public_job(job):
    internal_keys = {"output_dir", "log_dir", "log_path", "summary_path", "resource_start", "resource_end"}
    return {key: value for key, value in job.items() if key not in internal_keys}


def _set_job(job_id, **updates):
    with jobs_lock:
        job = jobs.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()
        output_dir = _job_dir(job_id)
        MucoService._write_json(output_dir / "status.json", _public_job(job))


def _run_job(job_id, request):
    try:
        log_path = _log_dir(job_id) / "muco.log"
        MucoService._append_log(log_path, "[%s] job %s started" % (time.strftime("%Y-%m-%d %H:%M:%S"), job_id))
        _set_job(job_id, status="running", stage="initializing", resource_start=_resource_snapshot())

        def progress(stage, stage1_done, stage1_total, stage2_done, stage2_total, stage3_done, stage3_total):
            _set_job(
                job_id,
                status="running" if stage != "done" else "done",
                stage=stage,
                stage1={"done": stage1_done, "total": stage1_total},
                stage2={"done": stage2_done, "total": stage2_total},
                stage3={"done": stage3_done, "total": stage3_total},
            )

        with open(str(log_path), "a") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                summary_path = service.generate(request, _job_dir(job_id) / "output", progress_callback=progress)
        _set_job(job_id, status="done", stage="done", resource_end=_resource_snapshot())
    except Exception as exc:
        _set_job(job_id, status="failed", stage="failed", error=str(exc), resource_end=_resource_snapshot())


@app.on_event("startup")
def startup():
    settings.job_root.mkdir(parents=True, exist_ok=True)
    settings.log_root.mkdir(parents=True, exist_ok=True)
    service.load()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    loaded = service.backbone_sampler is not None and service.sidechain_sampler is not None
    return {"ready": loaded, "cuda_available": torch.cuda.is_available(), "device": service.device}


@app.get("/resources")
def resources():
    return _resource_snapshot()


@app.post("/jobs", response_model=MucoJobResponse)
def create_job(request: MucoJobRequest):
    try:
        request.sequence = MucoService._validate_sequence(request.sequence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if request.K > settings.max_k:
        raise HTTPException(status_code=400, detail="K must be <= %d" % settings.max_k)
    if request.M > settings.max_m:
        raise HTTPException(status_code=400, detail="M must be <= %d" % settings.max_m)
    if request.relax_platform not in {"CUDA", "CPU", "OpenCL"}:
        raise HTTPException(status_code=400, detail="relax_platform must be CUDA, CPU, or OpenCL")
    if request.make_zip and not settings.download_enabled:
        raise HTTPException(status_code=400, detail="Downloads are disabled on this server")

    job_id = time.strftime("%Y%m%dT%H%M%S_") + str(uuid.uuid4())
    job_dir = _job_dir(job_id)
    log_dir = _log_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    MucoService._write_json(job_dir / "request.json", _model_dump(request))
    _set_job(job_id, status="queued", stage="queued", created_at=time.time(), output_dir=str(job_dir), log_dir=str(log_dir))
    executor.submit(_run_job, job_id, request)
    return MucoJobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    status_path = _job_dir(job_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with open(str(status_path), "r") as f:
        return json.load(f)


@app.get("/jobs/{job_id}/summary")
def get_summary(job_id: str):
    summary_path = _job_dir(job_id) / "output" / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    with open(str(summary_path), "r") as f:
        return json.load(f)


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    output_dir = _job_dir(job_id) / "output"
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    zip_files = sorted(output_dir.glob("*.zip"))
    if not zip_files:
        raise HTTPException(status_code=404, detail="Zip not found")
    return FileResponse(str(zip_files[0]), filename=zip_files[0].name)


@app.get("/jobs/{job_id}/files")
def get_file(job_id: str, path: str = Query(...)):
    root = _job_dir(job_id).resolve()
    if path.startswith("pdb_") or path.startswith("png_"):
        file_path = _file_path_from_token(job_id, path).resolve()
    else:
        file_path = Path(path).resolve()
    if not str(file_path).startswith(str(root) + os.sep):
        raise HTTPException(status_code=400, detail="File is outside job output directory")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=file_path.name)


def _file_path_from_token(job_id, token):
    parts = token.split("_")
    if len(parts) != 3 or parts[0] not in {"pdb", "png"}:
        raise HTTPException(status_code=400, detail="Invalid file token")
    try:
        k = int(parts[1])
        m = int(parts[2])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file token")

    summary_path = _job_dir(job_id) / "output" / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    with open(str(summary_path), "r") as f:
        rows = json.load(f)
    row = next((item for item in rows if item.get("k") == k and item.get("m") == m), None)
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    selected = row.get("render_png") if parts[0] == "png" else row.get("relaxed_pdb") or row.get("sidechain_pdb")
    if not selected:
        raise HTTPException(status_code=404, detail="File not found")
    return Path(selected)


@app.get("/jobs/{job_id}/logs")
def get_logs(job_id: str):
    log_path = _log_dir(job_id) / "muco.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(str(log_path), media_type="text/plain", filename="muco.log")


@app.post("/feedback")
def feedback(payload: Dict[str, object]):
    settings.log_root.mkdir(parents=True, exist_ok=True)
    feedback_path = settings.log_root / "feedback.jsonl"
    MucoService._append_log(feedback_path, json.dumps({"created_at": time.time(), "payload": payload}))
    return {"ok": True}
