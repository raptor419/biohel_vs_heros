#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_fold_row(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL summary across CV and seeds (HEROS-like)")
    parser.add_argument("--o", dest="outputPath", type=str, required=True,
                        help="Path to BioHEL output root: .../output/BioHEL_<name>")
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)
    parser.add_argument("--plots", dest="plots", action="store_true",
                        help="Generate boxplots")

    options = parser.parse_args(argv[1:])
    outputPath = Path(options.outputPath)
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds
    make_plots = options.plots

    # Expected structure:
    # outputPath/<dataset>/seed_i/cv_j/result_row.csv

    # -----------------------------
    # 1) Per-seed CV summaries
    # -----------------------------
    for dataset in os.listdir(outputPath):
        data_level_path = outputPath / dataset
        if not data_level_path.is_dir():
            continue

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            if not seed_level_path.exists():
                continue

            fold_dfs: List[pd.DataFrame] = []
            for j in range(1, cv_partitions + 1):
                cv_level_path = seed_level_path / f"cv_{j}"
                row_path = cv_level_path / "result_row.csv"
                if not row_path.exists():
                    continue
                fold_dfs.append(read_fold_row(row_path))

            if not fold_dfs:
                continue

            all_folds = pd.concat(fold_dfs, ignore_index=True)

            # numeric columns to aggregate
            metrics_cols = ["train_accuracy", "test_accuracy", "num_rules", "runtime", "wall_time"]
            present_cols = [c for c in metrics_cols if c in all_folds.columns]

            mean_df = all_folds[present_cols].mean(numeric_only=True).to_frame().T
            sd_df = all_folds[present_cols].std(numeric_only=True).to_frame().T

            # preserve identifiers
            mean_df.insert(0, "Dataset", dataset)
            mean_df.insert(1, "Seed", i)
            sd_df.insert(0, "Dataset", dataset)
            sd_df.insert(1, "Seed", i)

            mean_df.to_csv(seed_level_path / "mean_CV_results.csv", index=False)
            sd_df.to_csv(seed_level_path / "sd_CV_results.csv", index=False)

    # -----------------------------
    # 2) Per-dataset seed summaries
    # -----------------------------
    for dataset in os.listdir(outputPath):
        data_level_path = outputPath / dataset
        if not data_level_path.is_dir():
            continue

        seed_mean_dfs: List[pd.DataFrame] = []
        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            mean_path = seed_level_path / "mean_CV_results.csv"
            if mean_path.exists():
                seed_mean_dfs.append(pd.read_csv(mean_path))

        if not seed_mean_dfs:
            continue

        all_seed_means = pd.concat(seed_mean_dfs, ignore_index=True)

        metrics_cols = [c for c in all_seed_means.columns if c not in ["Dataset", "Seed"]]
        mean_seed = all_seed_means[metrics_cols].mean(numeric_only=True).to_frame().T
        sd_seed = all_seed_means[metrics_cols].std(numeric_only=True).to_frame().T

        mean_seed.insert(0, "Dataset", dataset)
        sd_seed.insert(0, "Dataset", dataset)

        mean_seed.to_csv(data_level_path / "mean_seed_results.csv", index=False)
        sd_seed.to_csv(data_level_path / "sd_seed_results.csv", index=False)

    # -----------------------------
    # 3) Global tables (all runs)
    # -----------------------------
    all_runs_rows: List[pd.DataFrame] = []
    cv_ave_rows: List[pd.DataFrame] = []

    for dataset in os.listdir(outputPath):
        data_level_path = outputPath / dataset
        if not data_level_path.is_dir():
            continue

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            if not seed_level_path.exists():
                continue

            # All folds for this seed
            for j in range(1, cv_partitions + 1):
                cv_level_path = seed_level_path / f"cv_{j}"
                row_path = cv_level_path / "result_row.csv"
                if row_path.exists():
                    df = pd.read_csv(row_path)
                    df.insert(0, "Dataset", dataset)
                    df.insert(1, "Seed", i)
                    df.insert(2, "CV", j)
                    all_runs_rows.append(df)

            # CV-averaged per seed
            mean_path = seed_level_path / "mean_CV_results.csv"
            if mean_path.exists():
                dfm = pd.read_csv(mean_path)
                cv_ave_rows.append(dfm)

    if all_runs_rows:
        all_runs = pd.concat(all_runs_rows, ignore_index=True)
        all_runs.to_csv(outputPath / "all_runs.csv", index=False)

    if cv_ave_rows:
        cv_ave = pd.concat(cv_ave_rows, ignore_index=True)
        cv_ave.to_csv(outputPath / "cv_ave_runs.csv", index=False)

    # -----------------------------
    # 4) Boxplots per dataset (optional)
    # -----------------------------
    if make_plots and all_runs_rows:
        # per dataset plots
        for dataset in os.listdir(outputPath):
            data_level_path = outputPath / dataset
            if not data_level_path.is_dir():
                continue

            df_ds = all_runs[all_runs["Dataset"] == dataset].copy()
            if df_ds.empty:
                continue

            # Test accuracy
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_ds, x="CV", y="test_accuracy")
            plt.title(f"{dataset}: Test Accuracy by CV (all seeds)")
            plt.xlabel("CV Fold")
            plt.ylabel("Test Accuracy")
            plt.savefig(data_level_path / "boxplot_test_accuracy_all.png", bbox_inches="tight")
            plt.close()

            # Rule count
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_ds, x="CV", y="num_rules")
            plt.title(f"{dataset}: Rule Count by CV (all seeds)")
            plt.xlabel("CV Fold")
            plt.ylabel("Rule Count")
            plt.savefig(data_level_path / "boxplot_rule_count_all.png", bbox_inches="tight")
            plt.close()

    print("BioHEL summary complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
