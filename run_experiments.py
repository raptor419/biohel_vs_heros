# run_experiments.py
from __future__ import annotations

import os
import re
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heros_eval import run_heros_for_dataset


# -----------------------------
# Paths / setup
# -----------------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "biohel_data"
DATA_DIR.mkdir(exist_ok=True)


# -----------------------------
# Dataset configs
# -----------------------------
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MUX6": {
        "train_path": "datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Train_1.txt",
        "test_path": "datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "description": "6-bit Multiplexer",
        "category": "Multiplexer",
        "features": 6,
    },
    "MUX11": {
        "train_path": "datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Train_1.txt",
        "test_path": "datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "description": "11-bit Multiplexer",
        "category": "Multiplexer",
        "features": 11,
    },
    "MUX20": {
        "train_path": "datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Train_1.txt",
        "test_path": "datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "description": "20-bit Multiplexer",
        "category": "Multiplexer",
        "features": 20,
    },
    "GAM_A": {
        "train_path": "datasets/gametes/A_uni_4add_CV_Train_1.txt",
        "test_path": "datasets/gametes/A_uni_4add_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "description": "GAMETES 4 Additive Univariate",
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_C": {
        "train_path": "datasets/gametes/C_2way_epistasis_CV_Train_1.txt",
        "test_path": "datasets/gametes/C_2way_epistasis_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "description": "GAMETES 2-way Epistasis",
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_E": {
        "train_path": "datasets/gametes/E_uni_4het_CV_Train_1.txt",
        "test_path": "datasets/gametes/E_uni_4het_CV_Test_1.txt",
        "excluded_columns": ["Model", "InstanceID", "Class"],
        "description": "GAMETES 4 Heterogeneous Univariate",
        "category": "GAMETES",
        "features": 100,
    },
}

DATASETS = list(DATASET_CONFIGS.keys())


# -----------------------------
# Helpers
# -----------------------------
def load_dataset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config["train_path"], sep="\t")
    test_df = pd.read_csv(config["test_path"], sep="\t")

    feature_names = [c for c in train_df.columns if c not in config["excluded_columns"]]
    X_train = train_df[feature_names].values
    y_train = train_df["Class"].values
    X_test = test_df[feature_names].values
    y_test = test_df["Class"].values

    row_id = train_df["InstanceID"].values if "InstanceID" in train_df.columns else np.arange(len(train_df))
    return X_train, y_train, X_test, y_test, row_id, feature_names, train_df, test_df


def convert_to_arff(df: pd.DataFrame, feature_names: List[str], filename: Path, relation_name: str) -> None:
    filename = Path(filename)
    with open(filename, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for feat in feature_names:
            unique_values = sorted(df[feat].unique())
            # assumes discrete integer-valued features (as in your original code)
            value_str = ",".join(map(str, [int(v) for v in unique_values]))
            f.write(f"@ATTRIBUTE {feat} {{{value_str}}}\n")

        class_values = sorted(df["Class"].unique())
        class_str = ",".join(map(str, [int(v) for v in class_values]))
        f.write(f"@ATTRIBUTE Class {{{class_str}}}\n\n")

        f.write("@DATA\n")
        for _, row in df.iterrows():
            values = [str(int(row[feat])) for feat in feature_names]
            values.append(str(int(row["Class"])))
            f.write(",".join(values) + "\n")


def create_biohel_config(output_path: Path) -> None:
    config_content = """crossover operator 1px
default class major
fitness function mdl
initialization min classifiers 20
initialization max classifiers 20
iterations 50
mdl initial tl ratio 0.25
mdl iteration 10
mdl weight relax factor 0.90
pop size 500
prob crossover 0.6
prob individual mutation 0.6
prob one 0.75
selection algorithm tournamentwor
tournament size 4
windowing ilas 1
dump evolution stats
smart init
class wise init
coverage breakpoint 0.01
repetitions of rule learning 2
coverage ratio 0.90
kr hyperrect
num expressed attributes init 15
hyperrectangle uses list of attributes
prob generalize list 0.10
prob specialize list 0.10
expected number of attributes 10
random seed 42
"""
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(config_content)


# -----------------------------
# BioHEL runner (no docker)
# -----------------------------
class BioHELRunner:
    def __init__(self, biohel_path: Path | str, data_dir: Path):
        self.biohel_path = str(biohel_path)
        self.data_dir = Path(data_dir)

    def _check_executable(self) -> None:
        p = Path(self.biohel_path)
        if p.exists() and not os.access(str(p), os.X_OK):
            raise RuntimeError(f"BioHEL exists but is not executable: {p}")

    def run_biohel(self, dataset_name: str) -> Dict[str, Any] | None:
        self._check_executable()

        dataset_dir = self.data_dir / dataset_name
        required = ["config.conf", "train.arff", "test.arff"]
        for f in required:
            if not (dataset_dir / f).exists():
                print(f"Missing {dataset_dir / f}")
                return None

        print(f"Running BioHEL on {dataset_name}...")
        start_time = time.time()

        cmd = [
            self.biohel_path,
            str(dataset_dir / "config.conf"),
            str(dataset_dir / "train.arff"),
            str(dataset_dir / "test.arff"),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(dataset_dir),
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - start_time

        # persist logs per dataset for debugging
        (dataset_dir / "biohel_stdout.txt").write_text(result.stdout)
        (dataset_dir / "biohel_stderr.txt").write_text(result.stderr)

        if result.returncode != 0:
            print(f"BioHEL failed for {dataset_name} (code {result.returncode}).")
            print(result.stderr[:2000])
            return None

        return self.parse_output(result.stdout, elapsed)

    def parse_output(self, output: str, wall_time: float) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "wall_time": float(wall_time),
            "test_accuracy": None,
            "train_accuracy": None,
            "num_rules": 0,
            "runtime": None,
            "rules": [],
        }

        for line in output.splitlines():
            if "Train accuracy :" in line:
                m = re.search(r"Train accuracy\s*:\s*([\d.]+)", line)
                if m:
                    results["train_accuracy"] = float(m.group(1))
            if "Test accuracy :" in line:
                m = re.search(r"Test accuracy\s*:\s*([\d.]+)", line)
                if m:
                    results["test_accuracy"] = float(m.group(1))
            if "Total time:" in line:
                m = re.search(r"Total time:\s*([\d.]+)", line)
                if m:
                    results["runtime"] = float(m.group(1))

        # phenotype / rules extraction
        in_phenotype = False
        for line in output.splitlines():
            if line.startswith("Phenotype:"):
                in_phenotype = True
                continue
            if in_phenotype:
                if line.strip() and not line.startswith("Train"):
                    results["rules"].append(line.strip())
                    results["num_rules"] += 1
                else:
                    break

        return results


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Set the BioHEL binary path:
    # - If you compiled it in repo root, keep as BASE_DIR / "biohel"
    # - If installed elsewhere, use an absolute path like "/usr/local/bin/biohel"
    BIOHEL_BIN = BASE_DIR / "biohel"

    biohel_runner = BioHELRunner(BIOHEL_BIN, DATA_DIR)
    print(f"Using BioHEL binary: {BIOHEL_BIN}")

    # 1) Prepare datasets (ARFF/config), run HEROS, save HEROS pickle per dataset
    heros_results: Dict[str, Dict[str, Any]] = {}

    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\nPreparing + HEROS: {dataset_name}")

        dataset_dir = DATA_DIR / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        X_train, y_train, X_test, y_test, row_id, feature_names, train_df, test_df = load_dataset(config)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # inputs for BioHEL
        convert_to_arff(train_df, feature_names, dataset_dir / "train.arff", f"{dataset_name}_Train")
        convert_to_arff(test_df, feature_names, dataset_dir / "test.arff", f"{dataset_name}_Test")
        create_biohel_config(dataset_dir / "config.conf")

        # HEROS metrics + pickle saved by heros_eval.py
        metrics = run_heros_for_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            row_id=row_id,
            random_state=42,
            verbose=True,
            save_pickle=True,
        )
        heros_results[dataset_name] = metrics

        print(
            f"HEROS Acc: {metrics['test_accuracy']:.4f}, "
            f"Rules: {metrics['num_rules']}, "
            f"Time: {metrics['training_time']:.1f}s"
        )

    print("\nHEROS done")

    # 2) Run BioHEL on all datasets
    biohel_results: Dict[str, Dict[str, Any]] = {}
    for dataset_name in DATASETS:
        print(f"\nBioHEL: {dataset_name}")
        result = biohel_runner.run_biohel(dataset_name)
        if result:
            biohel_results[dataset_name] = result
            ta = result["test_accuracy"]
            nr = result["num_rules"]
            rt = result["runtime"]
            if ta is not None and rt is not None:
                print(f"BioHEL Acc: {ta:.4f}, Rules: {nr}, Time: {rt:.1f}s")
            else:
                print(f"BioHEL completed; parsed metrics missing. Rules: {nr}")
        else:
            print("BioHEL failed.")

    print("\nBioHEL done")

    # 3) Build comparison dataframe
    comparison_rows: List[Dict[str, Any]] = []

    for dataset_name, config in DATASET_CONFIGS.items():
        if dataset_name in heros_results:
            h = heros_results[dataset_name]
            comparison_rows.append(
                {
                    "Dataset": dataset_name,
                    "Category": config["category"],
                    "Features": config["features"],
                    "Algorithm": "HEROS",
                    "Test Accuracy": h["test_accuracy"],
                    "Train Accuracy": h.get("train_accuracy"),
                    "Num Rules": h["num_rules"],
                    "Training Time (s)": h["training_time"],
                }
            )

        if dataset_name in biohel_results:
            b = biohel_results[dataset_name]
            comparison_rows.append(
                {
                    "Dataset": dataset_name,
                    "Category": config["category"],
                    "Features": config["features"],
                    "Algorithm": "BioHEL",
                    "Test Accuracy": b["test_accuracy"],
                    "Train Accuracy": b["train_accuracy"],
                    "Num Rules": b["num_rules"],
                    "Training Time (s)": b["runtime"],
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)

    # 4) Console summary
    print("\nComparison Results")
    print("-" * 90)
    print(f"{'Dataset':<10} {'Category':<12} {'HEROS Acc':<12} {'BioHEL Acc':<12} {'HEROS Rules':<12} {'BioHEL Rules':<12}")
    print("-" * 90)

    for dataset_name in DATASETS:
        config = DATASET_CONFIGS[dataset_name]
        h_row = comparison_df[(comparison_df["Dataset"] == dataset_name) & (comparison_df["Algorithm"] == "HEROS")]
        b_row = comparison_df[(comparison_df["Dataset"] == dataset_name) & (comparison_df["Algorithm"] == "BioHEL")]

        h_acc = f"{h_row['Test Accuracy'].values[0]:.1%}" if len(h_row) else "N/A"
        b_acc = f"{b_row['Test Accuracy'].values[0]:.1%}" if len(b_row) else "N/A"
        h_rules = int(h_row["Num Rules"].values[0]) if len(h_row) else "N/A"
        b_rules = int(b_row["Num Rules"].values[0]) if len(b_row) else "N/A"

        print(f"{dataset_name:<10} {config['category']:<12} {h_acc:<12} {b_acc:<12} {h_rules:<12} {b_rules:<12}")

    # 5) Plots
    plot_dir = DATA_DIR / "comparison_plots"
    plot_dir.mkdir(exist_ok=True)

    x = np.arange(len(DATASETS))
    width = 0.35

    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, algo in enumerate(["HEROS", "BioHEL"]):
        accs = []
        for d in DATASETS:
            sub = comparison_df[(comparison_df["Dataset"] == d) & (comparison_df["Algorithm"] == algo)]
            accs.append(float(sub["Test Accuracy"].values[0]) * 100 if len(sub) else 0.0)

        ax.bar(x + width * (i - 0.5), accs, width, label=algo)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("HEROS vs BioHEL Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy_comparison.png", dpi=300)
    plt.close(fig)

    # Rules comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, algo in enumerate(["HEROS", "BioHEL"]):
        rules = []
        for d in DATASETS:
            sub = comparison_df[(comparison_df["Dataset"] == d) & (comparison_df["Algorithm"] == algo)]
            rules.append(int(sub["Num Rules"].values[0]) if len(sub) else 0)

        ax.bar(x + width * (i - 0.5), rules, width, label=algo)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Rules")
    ax.set_title("HEROS vs BioHEL Rules")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "rules_comparison.png", dpi=300)
    plt.close(fig)

    # 6) CSV-only outputs
    out_dir = DATA_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)

    comparison_csv = out_dir / "comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nSaved: {comparison_csv}")

    pivot = comparison_df.pivot_table(
        index=["Dataset", "Category", "Features"],
        columns="Algorithm",
        values=["Test Accuracy", "Num Rules", "Training Time (s)"],
        aggfunc="first",
    )
    pivot_csv = out_dir / "side_by_side.csv"
    pivot.reset_index().to_csv(pivot_csv, index=False)
    print(f"Saved: {pivot_csv}")

    rules_dir = out_dir / "biohel_rules"
    rules_dir.mkdir(exist_ok=True)

    for dataset_name, results in biohel_results.items():
        rules = results.get("rules", [])
        if rules:
            rules_df = pd.DataFrame({"Rule": rules})
            rules_csv = rules_dir / f"{dataset_name}_biohel_rules.csv"
            rules_df.to_csv(rules_csv, index=False)
            print(f"Saved: {rules_csv}")

    # Optional: reproducibility/debug JSON (remove if you want strictly CSV+plots only)
    raw_json = out_dir / "raw_results.json"
    with open(raw_json, "w") as f:
        json.dump({"heros": heros_results, "biohel": biohel_results}, f, indent=2)
    print(f"Saved: {raw_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
