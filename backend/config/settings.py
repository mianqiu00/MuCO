import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _path_env(name, default):
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _int_env(name, default):
    return int(os.environ.get(name, str(default)))


def _bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    job_root: Path = _path_env("JOB_ROOT", "/data/muco/jobs")
    log_root: Path = _path_env("LOG_ROOT", "/data/muco/logs")
    max_k: int = _int_env("MAX_K", 3)
    max_m: int = _int_env("MAX_M", 5)
    download_enabled: bool = _bool_env("DOWNLOAD_ENABLED", True)
    redis_url: str = os.environ.get("REDIS_URL", "")
    model_path: Path = _path_env("MODEL_PATH", ROOT / "params")
    device: int = _int_env("MUCO_DEVICE", 0)
    seed: int = _int_env("MUCO_SEED", 42)
    backbone_steps: int = _int_env("MUCO_BACKBONE_STEPS", 100)
    sidechain_steps: int = _int_env("MUCO_SIDECHAIN_STEPS", 10)
    sidechain_coeff: float = float(os.environ.get("MUCO_SIDECHAIN_COEFF", "5.0"))
    noise_scale: float = float(os.environ.get("MUCO_NOISE_SCALE", "1.0"))
    min_t: float = float(os.environ.get("MUCO_MIN_T", "0.01"))
    use_gt_masks: bool = _bool_env("MUCO_USE_GT_MASKS", False)


settings = Settings()
