#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import re
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def build_feature_names(df: pd.DataFrame, outcome_label: str, instanceID_label: str, excluded_column: str) -> List[str]:
    cols = list(df.columns)
    for c in [excluded_column, instanceID_label]:
        if c in cols:
            cols.remove(c)
    if outcome_label in cols:
        cols.remove(outcome_label)
    return cols


def convert_to_arff(df: pd.DataFrame, feature_names: List[str], outcome_label: str, filename: Path, relation_name: str) -> None:
    with open(filename, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for feat in feature_names:
            unique_values = sorted(df[feat].unique())
            # BioHEL expects discrete values; force ints
            value_str = ",".join(map(str, [int(v) for v in unique_values]))
            f.write(f"@ATTRIBUTE {feat} {{{value_str}}}\n")

        class_values = sorted(df[outcome_label].unique())
        class_str = ",".join(map(str, [int(v) for v in class_values]))
        f.write(f"@ATTRIBUTE {outcome_label} {{{class_str}}}\n\n")

        f.write("@DATA\n")
        for _, row in df.iterrows():
            vals = [str(int(row[feat])) for feat in feature_names]
            vals.append(str(int(row[outcome_label])))
            f.write(",".join(vals) + "\n")


def create_biohel_config(output_path: Path, seed: int) -> None:
    # Same defaults you used previously, but inject the seed.
    # BioHEL config uses "random seed <int>"
    config_content = f"""crossover operator 1px
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
random seed {seed}
"""
    output_path.write_text(config_content)


def parse_biohel_output(output: str, wall_time: float) -> Dict[str, Any]:
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

    # Phenotype rules
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


def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL HPC Job: one CV fold + one seed")
    parser.add_argument("--d", dest="full_data_path", type=str, required=True,
                        help="Path to train fold file: <name>_CV_Train_k.txt")
    parser.add_argument("--o", dest="outputPath", type=str, required=True,
                        help="Output directory for this fold")
    parser.add_argument("--biohel", dest="biohel_bin", type=str, required=True,
                        help="BioHEL binary path or command on PATH")

    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")

    parser.add_argument("--rs", dest="random_state", type=int, default=42)
    parser.add_argument("--v", dest="verbose", action="store_true")

    options = parser.parse_args(argv[1:])

    full_data_path = options.full_data_path
    outputPath = Path(options.outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)

    biohel_bin = options.biohel_bin
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    seed = options.random_state

    # Derive test fold path like HEROS did
    test_data_path = full_data_path.replace("Train", "Test")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Expected test fold at: {test_data_path}")

    # Load
    train_df = load_df(full_data_path)
    test_df = load_df(test_data_path)

    # Determine features from train
    feature_names = build_feature_names(train_df, outcome_label, instanceID_label, excluded_column)

    # Build ARFF + config inside output folder
    train_arff = outputPath / "train.arff"
    test_arff = outputPath / "test.arff"
    conf_path = outputPath / "config.conf"

    convert_to_arff(train_df, feature_names, outcome_label, train_arff, relation_name="Train")
    convert_to_arff(test_df, feature_names, outcome_label, test_arff, relation_name="Test")
    create_biohel_config(conf_path, seed=seed)

    # Run BioHEL
    cmd = [biohel_bin, str(conf_path), str(train_arff), str(test_arff)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(outputPath), capture_output=True, text=True)
    wall_time = time.time() - start

    (outputPath / "biohel_stdout.txt").write_text(proc.stdout)
    (outputPath / "biohel_stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"BioHEL failed (code {proc.returncode}). See biohel_stderr.txt")

    metrics = parse_biohel_output(proc.stdout, wall_time)

    # Save metrics
    with open(outputPath / "metrics.json", "w") as f:
        json.dump(
            {
                "train_file": full_data_path,
                "test_file": test_data_path,
                "seed": seed,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    # Save rules CSV if present
    if metrics.get("rules"):
        pd.DataFrame({"Rule": metrics["rules"]}).to_csv(outputPath / "rules.csv", index=False)

    # Sentinel row CSV (main uses this to check completion)
    row = {
        "train_file": full_data_path,
        "test_file": test_data_path,
        "seed": seed,
        "train_accuracy": metrics.get("train_accuracy"),
        "test_accuracy": metrics.get("test_accuracy"),
        "num_rules": metrics.get("num_rules"),
        "runtime": metrics.get("runtime"),
        "wall_time": metrics.get("wall_time"),
    }
    pd.DataFrame([row]).to_csv(outputPath / "result_row.csv", index=False)

    if options.verbose:
        print(f"Done. seed={seed} test_acc={row['test_accuracy']} rules={row['num_rules']}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
