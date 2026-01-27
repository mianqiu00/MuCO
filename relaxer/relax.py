#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from auto import ForceFieldMinimizerAuto

INPUT_ROOT = "./unrelaxed"
OUTPUT_ROOT = "./relaxed"
PLATFORM = "CUDA"

GPU_IDS = [0, 1, 2, 3]
WORKERS_PER_GPU = 4
TOTAL_WORKERS = len(GPU_IDS) * WORKERS_PER_GPU


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_pdb_files(root):
    tasks = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdb"):
                abs_path = os.path.join(dirpath, fn)
                rel_dir = os.path.relpath(dirpath, root)
                tasks.append((abs_path, rel_dir, fn))
    return tasks


def process_one(pdb_path, rel_dir, filename, gpu_queue):
    """
    Worker 进程函数
    """
    gpu_id = None
    try:
        gpu_id = gpu_queue.get()
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        minimizer = ForceFieldMinimizerAuto(
            platform=PLATFORM,
            log=False
        )

        out_dir = os.path.join(OUTPUT_ROOT, rel_dir)
        ensure_dir(out_dir)
        out_pdb = os.path.join(out_dir, filename)

        ret = minimizer(pdb_path, out_pdb)

        result = {
            "filename": filename,
            "mode": ret["mode"],
            "bond_atom_head": ret["bond_atoms"][0],
            "bond_atom_tail": ret["bond_atoms"][1],
            "distance_pre": ret["distance_pre"],
            "distance_post": ret["distance_post"],
            "cyclized": ret["cyclized"],
            "einit": ret["einit"],
            "efinal": ret["efinal"],
        }
        return rel_dir, result, None

    except Exception as e:
        return rel_dir, {
            "filename": filename,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, e
    
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)


def main():
    m = multiprocessing.Manager()
    gpu_queue = m.Queue()

    for gpu_id in GPU_IDS:
        for _ in range(WORKERS_PER_GPU):
            gpu_queue.put(gpu_id)

    pdb_tasks = collect_pdb_files(INPUT_ROOT)
    if not pdb_tasks:
        print("No PDB files found.")
        return

    print(f"Found {len(pdb_tasks)} PDB files.")
    print(f"Using GPUs: {GPU_IDS}")
    print(f"Workers per GPU: {WORKERS_PER_GPU}")
    print(f"Total Parallel Workers: {TOTAL_WORKERS}")

    summaries = {}

    with ProcessPoolExecutor(max_workers=TOTAL_WORKERS) as executor:
        futures = [
            executor.submit(process_one, pdb, rel_dir, fn, gpu_queue)
            for pdb, rel_dir, fn in pdb_tasks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Relaxing PDBs"):
            rel_dir, result, err = fut.result()
            summaries.setdefault(rel_dir, []).append(result)
            
            if err:
                print(f"[ERROR] {result['filename']} | {result.get('error')}")

    for rel_dir, records in summaries.items():
        out_dir = os.path.join(OUTPUT_ROOT, rel_dir)
        ensure_dir(out_dir)
        csv_path = os.path.join(out_dir, "summary.csv")

        fieldnames = [
            "filename", "mode", "bond_atom_head", "bond_atom_tail",
            "distance_pre", "distance_post", "cyclized", "einit", "efinal"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                if "error" in r:
                    continue
                writer.writerow(r)

    print("All done.")


if __name__ == "__main__":
    main()