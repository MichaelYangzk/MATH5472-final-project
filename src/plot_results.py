# src/plot_results.py

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import RESULTS_DIR


def plot_metric_bar(metric: str, out_name: str):
    """
    从 metrics.csv 中读取指定指标, 画成条形图。
    metric 可以是 'accuracy', 'f1', 'roc_auc' 等。
    """
    metrics_path = RESULTS_DIR / "metrics.csv"
    df = pd.read_csv(metrics_path)

    if metric not in df.columns:
        raise ValueError(f"Metric {metric} not found in metrics.csv")

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[metric])
    plt.ylabel(metric)
    plt.title(f"Model comparison on {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = RESULTS_DIR / out_name
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {metric} bar plot to {out_path}")


def main():
    plot_metric_bar("accuracy", "model_comparison_accuracy.png")
    plot_metric_bar("f1", "model_comparison_f1.png")
    plot_metric_bar("roc_auc", "model_comparison_roc_auc.png")


if __name__ == "__main__":
    main()
