# File: 07_plot_results.py
# Description:
# This script processes model evaluation metrics, aggregates results,
# and generates performance visualizations and summary tables.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === Setup Output Directories ===
plot_dir = "results/plots"
table_dir = "results/tables"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

# === Set Visual Theme ===
sns.set_style("whitegrid")

# === Load Data ===
df = pd.read_csv("results/metrics.csv", keep_default_na=False)
print("[INFO] Loaded metrics.csv")

# === Normalize Column Names ===
rename_map = {
    'F1 Score': 'F1', 'F1-Score': 'F1',
    'ROC AUC': 'ROC_AUC', 'ROC-AUC': 'ROC_AUC', 'AUC': 'ROC_AUC',
    'Precision Score': 'Precision', 'Recall Score': 'Recall'
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# === Convert Metric Columns ===
metrics = ['Precision', 'Recall', 'F1', 'ROC_AUC']
df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

# === Clean and Standardize Categorical Columns ===
for col in ['Model', 'Sampling', 'Weighting']:
    df[col] = df[col].astype(str).fillna("None")

# === Check for Missing Values ===
if df[metrics].isnull().values.any():
    print("Warning: Missing values detected in the metrics data. They will be handled in plots.")

# === Aggregate Across Runs ===
group_cols = ['Model', 'Sampling', 'Weighting']
agg_df = df.groupby(group_cols)[metrics].mean(numeric_only=True).reset_index()

# === Bar Plots ===
for metric in ['F1', 'ROC_AUC']:
    if metric not in agg_df.columns:
        continue

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=agg_df, x='Model', y=metric, hue='Sampling')
    ax.set_title(f"{metric} Score by Model and Sampling")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")

    for p in ax.patches:
        if not np.isnan(p.get_height()) and p.get_height() > 0:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8)

    ax.legend(title="Sampling", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    out_path = f"{plot_dir}/bar_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")

# === Heatmaps ===
for metric in ['F1', 'ROC_AUC']:
    pivot = agg_df.copy()
    pivot['Sampling_Weighting'] = pivot['Sampling'] + "_" + pivot['Weighting']
    heat_df = pivot.pivot(index='Model', columns='Sampling_Weighting', values=metric)

    if heat_df.dropna(how='all').empty:
        print(f"[Warning] Skipping heatmap for {metric} — no valid data available.")
        continue

    plt.figure(figsize=(max(8, 1.5 * len(heat_df.columns)), 5))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, linecolor='gray')
    plt.title(f"{metric} Score Heatmap")
    plt.xlabel("Sampling_Weighting")
    plt.ylabel("Model")
    plt.tight_layout()
    out_path = f"{plot_dir}/heatmap_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")

# === Radar Chart (Top 3 Models by F1) ===
top_metric = 'F1'
top_models = agg_df.dropna(subset=[top_metric]).nlargest(3, top_metric)

radar_metrics = ['Precision', 'Recall', 'F1', 'ROC_AUC']
top_models = top_models.dropna(subset=radar_metrics)

if not top_models.empty and len(top_models) >= 1:
    angles = [n / float(len(radar_metrics)) * 2 * np.pi for n in range(len(radar_metrics))]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for _, row in top_models.iterrows():
        values = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
        label = f"{row['Model']} ({row['Sampling']}, {row['Weighting']})"
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Top 3 Models Radar Chart", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    out_path = f"{plot_dir}/radar_top_models.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")
else:
    print("[Notice] Radar chart skipped due to insufficient metric columns or top entries.")

# === Export Tables ===
summary_path = f"{table_dir}/metrics_summary.csv"
agg_df.to_csv(summary_path, index=False)
print(f"[Saved] {summary_path}")

for metric in ['F1', 'ROC_AUC']:
    if metric in agg_df.columns:
        top5 = agg_df.dropna(subset=[metric]).nlargest(5, metric)
        top_path = f"{table_dir}/top_5_{metric}.csv"
        top5.to_csv(top_path, index=False)
        print(f"[Saved] {top_path}")

print("All plots and tables successfully generated.")
