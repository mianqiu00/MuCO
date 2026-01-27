#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import mdtraj as md
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

INPUT_ROOT = "./relaxed" 
N_WORKERS = 8     

def calc_ss_mdtraj(pdb_path, filename):
    """
    计算单个 PDB 的二级结构比例
    """
    try:
        traj = md.load(pdb_path)
        
        dssp = md.compute_dssp(traj, simplified=True)[0]
        
        total_res = len(dssp)
        if total_res == 0:
            return None

        n_helix = np.sum(dssp == 'H')
        n_sheet = np.sum(dssp == 'E')
        n_coil  = np.sum(dssp == 'C')
        
        return {
            "filename": filename,
            "length": total_res,
            "helix": n_helix / total_res,
            "sheet": n_sheet / total_res,
            "coil":  n_coil  / total_res
        }
        
    except Exception as e:
        return {
            "filename": filename,
            "length": 0,
            "helix": 0.0,
            "sheet": 0.0,
            "coil": 0.0,
            "error": str(e)
        }

def process_wrapper(args):
    dirpath, filename = args
    full_path = os.path.join(dirpath, filename)
    result = calc_ss_mdtraj(full_path, filename)
    return dirpath, result

def collect_files(root):
    tasks = []
    print(f"Scanning {root} for PDB files...")
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdb"):
                tasks.append((dirpath, fn))
    return tasks

def main():
    tasks = collect_files(INPUT_ROOT)
    if not tasks:
        print("No PDB files found.")
        return
    print(f"Found {len(tasks)} PDB files. Starting analysis with {N_WORKERS} workers...")

    results_by_folder = defaultdict(list)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_wrapper, task) for task in tasks]
        
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Calculating SS"):
            dirpath, res = fut.result()
            if res:
                results_by_folder[dirpath].append(res)

    print("\nWriting CSV files to each folder...")
    count_csv = 0
    
    for dirpath, rows in results_by_folder.items():
        if not rows:
            continue
            
        csv_path = os.path.join(dirpath, "structure.csv")
        
        fieldnames = ["filename", "length", "helix", "sheet", "coil"]
        
        rows.sort(key=lambda x: x['filename'])

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            count_csv += 1
        except Exception as e:
            print(f"[Error] Failed to write {csv_path}: {e}")

    print(f"Done. Generated {count_csv} structure.csv files.")

if __name__ == "__main__":
    main()