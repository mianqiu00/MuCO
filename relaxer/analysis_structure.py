#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_ROOT = "./relaxed"
OUTPUT_DIR = "./comparison_result"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_all_structures(root):
    all_dfs = []
    print(f"Scanning {root} for structure.csv files...")

    for dirpath, _, filenames in os.walk(root):
        if "structure.csv" in filenames:
            csv_path = os.path.join(dirpath, "structure.csv")
            try:
                df = pd.read_csv(csv_path)
                rel_dir = os.path.relpath(dirpath, root)
                df['folder'] = rel_dir
                all_dfs.append(df)
            except Exception as e:
                print(f"[WARN] Failed to read {csv_path}: {e}")

    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)

def main():
    ensure_dir(OUTPUT_DIR)

    df = load_all_structures(INPUT_ROOT)
    if df.empty:
        print("No structure data found. Did you run the previous script?")
        return

    print("\nCalculating statistics...")
    stats = df.groupby('folder')[['length', 'helix', 'sheet', 'coil']].agg(['mean', 'std'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.sort_values(by='helix_mean', ascending=False)
    
    csv_out = os.path.join(OUTPUT_DIR, "structure_comparison_stats.csv")
    stats.to_csv(csv_out)
    print(f"[Saved] Stats report: {csv_out}")
    
    print("\n=== Average Secondary Structure Content ===")
    print(stats[['helix_mean', 'sheet_mean', 'coil_mean']].to_string())

    print("\nGenerating plots...")
    sns.set_theme(style="whitegrid")
    
    folder_order = stats.index.tolist()
    n_folders = len(folder_order)
    
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Helix
    sns.boxplot(data=df, x='folder', y='helix', order=folder_order, ax=axes[0], palette="Reds")
    axes[0].set_title("Helix Ratio Distribution")
    axes[0].set_ylabel("Ratio (0.0 - 1.0)")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Sheet
    sns.boxplot(data=df, x='folder', y='sheet', order=folder_order, ax=axes[1], palette="Blues")
    axes[1].set_title("Sheet Ratio Distribution")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Coil
    sns.boxplot(data=df, x='folder', y='coil', order=folder_order, ax=axes[2], palette="Greens")
    axes[2].set_title("Coil Ratio Distribution")
    axes[2].set_ylabel("")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "structure_distribution_boxplot.png")
    plt.savefig(plot1_path, dpi=300)
    print(f"[Saved] Boxplot: {plot1_path}")

    mean_df = df.groupby('folder')[['helix', 'sheet', 'coil']].mean()
    mean_df = mean_df.reindex(folder_order)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    mean_df.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff', '#99ff99'], ax=ax2, width=0.8)
    
    ax2.set_title("Average Secondary Structure Composition")
    ax2.set_ylabel("Proportion")
    ax2.set_ylim(0, 1.0)
    ax2.legend(["Helix", "Sheet", "Coil"], loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "structure_composition_stacked.png")
    plt.savefig(plot2_path, dpi=300)
    print(f"[Saved] Stacked Plot: {plot2_path}")

if __name__ == "__main__":
    main()