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

def load_data(root):
    all_data = []
    print(f"Scanning {root}...")

    for dirpath, _, filenames in os.walk(root):
        if "summary.csv" in filenames:
            csv_path = os.path.join(dirpath, "summary.csv")
            try:
                df = pd.read_csv(csv_path)
                rel_dir = os.path.relpath(dirpath, root)
                df['folder'] = rel_dir
                all_data.append(df)
            except Exception as e:
                print(f"[WARN] Error reading {csv_path}: {e}")

    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def main():
    ensure_dir(OUTPUT_DIR)

    # 1. 加载数据
    df = load_data(INPUT_ROOT)
    if df.empty:
        print("No data found.")
        return

    df['cyclized'] = df['cyclized'].astype(str).str.lower() == 'true'
    df['efinal'] = pd.to_numeric(df['efinal'], errors='coerce')
    
    count_before = df['cyclized'].sum()
    
    high_energy_mask = df['efinal'] > 1000
    df.loc[high_energy_mask, 'cyclized'] = False
    
    count_after = df['cyclized'].sum()
    filtered_count = count_before - count_after
    
    if filtered_count > 0:
        print(f"\n[Filter Applied] {filtered_count} structures were marked as failed due to High Energy (>1000 kJ/mol).")

    df['delta_energy'] = df['einit'] - df['efinal']

    folder_stats = df.groupby('folder')['cyclized'].agg(['count', 'sum']).rename(columns={'count': 'Total', 'sum': 'Success'})
    folder_stats['Success_Rate'] = folder_stats['Success'] / folder_stats['Total']

    df_cyclized = df[df['cyclized']].copy()

    if df_cyclized.empty:
        print("No cyclized structures found (after filtering high energy ones).")
        return

    energy_stats = df_cyclized.groupby('folder').agg({
        'efinal': ['mean', 'std'],
        'delta_energy': ['mean', 'std'],
        'distance_post': ['mean']
    })
    energy_stats.columns = ['_'.join(col).strip() for col in energy_stats.columns.values]

    mode_counts = pd.crosstab(df_cyclized['folder'], df_cyclized['mode'])
    mode_ratios = mode_counts.div(mode_counts.sum(axis=1), axis=0)
    
    mode_counts.columns = [f"Count_{c}" for c in mode_counts.columns]
    mode_ratios.columns = [f"Ratio_{c}" for c in mode_ratios.columns]

    final_report = folder_stats.join(energy_stats).join(mode_counts).join(mode_ratios)
    
    if 'delta_energy_mean' in final_report.columns:
        final_report = final_report.sort_values(by='delta_energy_mean', ascending=False)
    
    csv_path = os.path.join(OUTPUT_DIR, "folder_comparison_full.csv")
    final_report.to_csv(csv_path)
    print(f"\n[Saved] Full statistics: {csv_path}")

    print("\nGenerating comprehensive plots...")
    sns.set_theme(style="whitegrid")
    
    n_folders = len(final_report)
    fig_height = 10 + (n_folders * 0.5)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, fig_height), gridspec_kw={'height_ratios': [2, 2, 1, 1.5]})
    
    order_idx = final_report.index

    # --- Plot 1: Energy Drop (Boxplot) ---
    sns.boxplot(data=df_cyclized, x='delta_energy', y='folder', order=order_idx, ax=axes[0], palette="vlag")
    axes[0].set_title("1. Energy Drop Distribution (Higher = More Stabilized)")
    axes[0].set_xlabel("Delta Energy (kJ/mol)")

    # --- Plot 2: Final Energy (Boxplot) ---
    sns.boxplot(data=df_cyclized, x='efinal', y='folder', order=order_idx, ax=axes[1], palette="mako")
    axes[1].set_title("2. Final Energy Distribution (Left/Lower is Better)")
    axes[1].set_xlabel("Final Energy (kJ/mol)")

    # --- Plot 3: Success Rate (Barplot) ---
    sns.barplot(x=final_report['Success_Rate'], y=final_report.index, ax=axes[2], palette="Blues_d")
    axes[2].set_title("3. Cyclization Success Rate (Energy > 1000 considered failed)")
    axes[2].set_xlim(0, 1.0)

    # --- Plot 4: Mode Distribution (Stacked Bar Plot) ---
    ratio_data = final_report[[c for c in final_report.columns if c.startswith("Ratio_")]]
    ratio_data.columns = [c.replace("Ratio_", "") for c in ratio_data.columns]
    
    if not ratio_data.empty:
        ratio_data.plot(kind='barh', stacked=True, ax=axes[3], colormap='tab10', width=0.8)
        axes[3].invert_yaxis() 
        axes[3].set_title("4. Mode Distribution (Ratio of Cyclization Types)")
        axes[3].set_xlabel("Proportion (0.0 - 1.0)")
        axes[3].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Mode")
    else:
        axes[3].text(0.5, 0.5, "No cyclized data to plot mode distribution", ha='center')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "folder_comparison_viz.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] Visualization: {plot_path}")

if __name__ == "__main__":
    main()