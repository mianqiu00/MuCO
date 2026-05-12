import argparse
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path

import torch
from ml_collections import ConfigDict
from omegaconf import OmegaConf

from runner.backbone_sampler import BackboneSampler
from runner.sidechain_sampler import SidechainSampler
from utils import seed_everything


ROOT = Path(__file__).resolve().parent
AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWYX")


def _read_json(path):
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {"samples": payload}
    if "samples" not in payload:
        return {"samples": [payload]}
    return payload


def _sample_entries(payload):
    entries = []
    for idx, item in enumerate(payload["samples"]):
        if isinstance(item, str):
            name = f"sample_{idx + 1}"
            sequence = item
        else:
            sequence = item.get("sequence") or item.get("seq")
            name = item.get("id") or item.get("name") or f"sample_{idx + 1}"
        if not sequence:
            raise ValueError(f"Missing sequence for JSON sample #{idx + 1}")
        sequence = sequence.strip().upper()
        bad = sorted(set(sequence) - AA_LETTERS)
        if bad:
            raise ValueError(f"Sample {name} contains unsupported residue letters: {''.join(bad)}")
        entries.append((str(name), sequence))
    return entries


def _load_backbone_config(args):
    if args.backbone_config:
        conf = OmegaConf.load(args.backbone_config)
    else:
        conf = OmegaConf.load(ROOT / "config" / "backbone.yaml")
    conf.experiment.use_gpu = torch.cuda.is_available() and args.device >= 0
    conf.experiment.batch_size = 1
    conf.experiment.eval_batch_size = 1
    conf.experiment.noise_scale = args.noise_scale
    conf.experiment.full_ckpt_dir = str(ROOT / "params" / "foldflow.pth")
    conf.data.num_t = args.backbone_steps
    conf.data.min_t = args.min_t
    conf.data.cache_full_dataset = False
    conf.data.cache_dataset_in_memory = False
    return conf


def _load_sidechain_config(args):
    ckpt_path = ROOT / "params" / "flowpacker.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing FlowPacker checkpoint: {ckpt_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("FlowPacker sampler currently requires CUDA because the existing model loader calls .cuda().")

    ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt_dict["config"]
    config.ckpt = str(ckpt_path)
    config.seed = args.seed
    config.direct_inference = True
    if getattr(config, "mode", None) == "flow":
        config.mode = getattr(config.train, "loss_type", "vf")
    config.conf_ckpt = getattr(config, "conf_ckpt", None)
    if not hasattr(config, "sample"):
        config.sample = ConfigDict()
    config.sample.n_samples = 1
    config.sample.batch_size = 1
    config.sample.num_steps = args.sidechain_steps
    config.sample.coeff = args.sidechain_coeff
    config.data.test_path = str(args.work_dir / "sidechain_input")
    return config


def _copy_for_sidechain(backbone_pdb, sidechain_input_dir):
    sidechain_input_dir.mkdir(parents=True, exist_ok=True)
    target = sidechain_input_dir / Path(backbone_pdb).name
    shutil.copyfile(backbone_pdb, target)
    return target


def _relax_pdb(input_pdb, output_pdb, platform, log=False):
    import sys

    relaxer_dir = ROOT / "relaxer"
    sys.path.insert(0, str(relaxer_dir))
    try:
        from auto import ForceFieldMinimizerAuto

        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        return ForceFieldMinimizerAuto(platform=platform, log=log)(
            str(input_pdb), str(output_pdb), log=log
        )
    finally:
        if str(relaxer_dir) in sys.path:
            sys.path.remove(str(relaxer_dir))


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _write_progress(args, stage, stage1_done=0, stage1_total=0, stage2_done=0, stage2_total=0, stage3_done=0, stage3_total=0, status="running"):
    progress_path = getattr(args, "progress_json", None)
    if not progress_path:
        return
    payload = {
        "status": status,
        "stage": stage,
        "stage1": {"done": stage1_done, "total": stage1_total},
        "stage2": {"done": stage2_done, "total": stage2_total},
        "stage3": {"done": stage3_done, "total": stage3_total},
        "updated_at": time.time(),
    }
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _write_success_zip(args, results):
    if not getattr(args, "make_zip", False):
        return None
    seq_name = results[0]["sequence"] if results else "muco"
    safe_seq_name = "".join(ch for ch in seq_name if ch.isalnum() or ch in "-_") or "muco"
    zip_path = args.work_dir / f"{safe_seq_name}.zip"
    successful = []
    for row in results:
        relax = row.get("relax") or {}
        pdb = row.get("relaxed_pdb")
        if pdb and relax.get("cyclized", False) and os.path.exists(pdb):
            successful.append((row, pdb))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row, pdb in successful:
            safe_seq = "".join(ch for ch in row["sequence"] if ch.isalnum() or ch in "-_") or row["id"]
            zf.write(pdb, arcname=f"{safe_seq}_{row['k']}_{row['m']}.pdb")
    return str(zip_path)


def run(args):
    args.work_dir = Path(args.output).resolve()
    backbone_dir = args.work_dir / "stage1_backbone"
    sidechain_dir = args.work_dir / "stage2_sidechain"
    relaxed_dir = args.work_dir / "stage3_relaxed"
    sidechain_input_dir = args.work_dir / "sidechain_input"
    for path in [backbone_dir, sidechain_dir, relaxed_dir, sidechain_input_dir]:
        path.mkdir(parents=True, exist_ok=True)

    payload = _read_json(args.input_json)
    samples = _sample_entries(payload)
    k = int(payload.get("K", args.K))
    m = int(payload.get("M", args.M))
    if k < 1 or m < 1:
        raise ValueError("K and M must both be >= 1")
    _write_progress(args, "initializing", status="running")

    seed_everything(args.seed)
    backbone_conf = _load_backbone_config(args)
    backbone_sampler = BackboneSampler(
        conf=backbone_conf,
        ckpt_epoch=args.backbone_epoch,
        output_dir=str(backbone_dir),
        device_id=args.device,
        is_sample=True,
        batch_size=1,
    )

    sidechain_config = _load_sidechain_config(args)
    sidechain_sampler = SidechainSampler(sidechain_config, use_gt_masks=args.use_gt_masks)

    backbone_jobs = []
    for sample_name, sequence in samples:
        for k_idx in range(1, k + 1):
            backbone_jobs.append({
                "id": sample_name,
                "sequence": sequence,
                "k": k_idx,
                "run_name": f"{sample_name}_k{k_idx}",
            })

    total_backbones = len(backbone_jobs)
    total_sidechains = total_backbones * m
    _write_progress(args, "stage1", stage1_done=0, stage1_total=total_backbones, stage2_done=0, stage2_total=total_sidechains, stage3_done=0, stage3_total=total_sidechains)
    stage1_step_total = max(int(args.backbone_steps), 1)

    _sync_cuda()
    stage1_start = time.perf_counter()
    backbone_paths = backbone_sampler.sample_sequences_to_pdb(
        [job["sequence"] for job in backbone_jobs],
        [job["run_name"] for job in backbone_jobs],
        str(backbone_dir),
        progress_callback=lambda step, total: _write_progress(
            args,
            "stage1",
            stage1_done=step,
            stage1_total=total,
            stage2_done=0,
            stage2_total=int(args.sidechain_steps) * total_sidechains,
            stage3_done=0,
            stage3_total=total_sidechains,
        ),
    )
    _sync_cuda()
    stage1_batch_seconds = time.perf_counter() - stage1_start
    stage1_seconds_each = stage1_batch_seconds / max(len(backbone_jobs), 1)
    _write_progress(args, "stage2", stage1_done=stage1_step_total, stage1_total=stage1_step_total, stage2_done=0, stage2_total=int(args.sidechain_steps) * total_sidechains, stage3_done=0, stage3_total=total_sidechains)

    results = []
    stage2_done = 0
    stage2_step_done = 0
    stage3_done = 0
    for job, backbone_pdb in zip(backbone_jobs, backbone_paths):
        sample_name = job["id"]
        sequence = job["sequence"]
        k_idx = job["k"]
        sidechain_input = _copy_for_sidechain(backbone_pdb, sidechain_input_dir)
        for m_idx in range(1, m + 1):
            packed_name = f"{sample_name}_k{k_idx}_m{m_idx}.pdb"
            packed_pdb = sidechain_dir / packed_name
            relaxed_pdb = relaxed_dir / packed_name
            _sync_cuda()
            stage2_start = time.perf_counter()
            sidechain_sampler.sample_pdb_to_pdb(
                sidechain_input,
                packed_pdb,
                progress_callback=lambda step, total: _write_progress(
                    args,
                    "stage2",
                    stage1_done=stage1_step_total,
                    stage1_total=stage1_step_total,
                    stage2_done=stage2_step_done + step,
                    stage2_total=int(args.sidechain_steps) * total_sidechains,
                    stage3_done=stage3_done,
                    stage3_total=total_sidechains,
                ),
            )
            _sync_cuda()
            stage2_seconds = time.perf_counter() - stage2_start
            stage2_done += 1
            stage2_step_done += int(args.sidechain_steps)
            _write_progress(args, "stage3" if not args.skip_relax else "finalizing", stage1_done=stage1_step_total, stage1_total=stage1_step_total, stage2_done=stage2_step_done, stage2_total=int(args.sidechain_steps) * total_sidechains, stage3_done=stage3_done, stage3_total=total_sidechains)
            relax_info = None
            stage3_seconds = 0.0
            if not args.skip_relax:
                stage3_start = time.perf_counter()
                relax_info = _relax_pdb(packed_pdb, relaxed_pdb, args.relax_platform, log=args.relax_log)
                stage3_seconds = time.perf_counter() - stage3_start
                stage3_done += 1
                _write_progress(args, "stage3", stage1_done=stage1_step_total, stage1_total=stage1_step_total, stage2_done=stage2_step_done, stage2_total=int(args.sidechain_steps) * total_sidechains, stage3_done=stage3_done, stage3_total=total_sidechains)
            results.append({
                "id": sample_name,
                "sequence": sequence,
                "length": len(sequence),
                "k": k_idx,
                "m": m_idx,
                "stage1_seconds": stage1_seconds_each,
                "stage1_batch_seconds": stage1_batch_seconds,
                "stage1_batch_size": len(backbone_jobs),
                "stage2_seconds": stage2_seconds,
                "stage3_seconds": stage3_seconds,
                "total_seconds": stage1_seconds_each + stage2_seconds + stage3_seconds,
                "backbone_pdb": str(backbone_pdb),
                "sidechain_pdb": str(packed_pdb),
                "relaxed_pdb": None if args.skip_relax else str(relaxed_pdb),
                "relax": relax_info,
            })

    summary_path = args.work_dir / "summary.json"
    zip_path = _write_success_zip(args, results)
    if zip_path:
        for row in results:
            row["success_zip"] = zip_path
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    _write_progress(args, "done", stage1_done=stage1_step_total, stage1_total=stage1_step_total, stage2_done=stage2_step_done, stage2_total=int(args.sidechain_steps) * total_sidechains, stage3_done=stage3_done, stage3_total=total_sidechains, status="done")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run MuCO 3-stage inference from sequence JSON.")
    parser.add_argument("input_json", help="JSON with sequence(s), optional K and M.")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "muco_infer"), help="Output directory.")
    parser.add_argument("--K", type=int, default=1, help="Backbone samples per sequence.")
    parser.add_argument("--M", type=int, default=1, help="Sidechain samples per backbone.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id; ignored on CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone_config", default=None)
    parser.add_argument("--backbone_epoch", type=int, default=100)
    parser.add_argument("--backbone_steps", type=int, default=100)
    parser.add_argument("--min_t", type=float, default=0.01)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--sidechain_steps", type=int, default=10)
    parser.add_argument("--sidechain_coeff", type=float, default=5.0)
    parser.add_argument("--use_gt_masks", action="store_true")
    parser.add_argument("--skip_relax", action="store_true")
    parser.add_argument("--relax_platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL"])
    parser.add_argument("--relax_log", action="store_true")
    parser.add_argument("--progress_json", default=None)
    parser.add_argument("--make_zip", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary_path = run(args)
    print(f"MuCO inference complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
