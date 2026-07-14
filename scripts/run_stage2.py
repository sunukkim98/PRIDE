"""
Stage 2 Hyperparameter Sweep — lambda_power weight mode
=========================================================
Sweeps energy_r × energy_lambda for LightGCN backbone across 4 datasets.
Datasets are processed sequentially; within each dataset, up to N_GPUS
experiments run in parallel (one process per GPU slot).

12 runs per dataset (4 energy_r × 3 energy_lambda), 48 total.

Usage (from project root):
    python scripts/run_stage2.py [--gpus 0,1,2,3] [--dry-run]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR     = RESULTS_DIR / "stage2_logs"
DONE_DIR    = LOG_DIR / "done"
CSV_PATH    = RESULTS_DIR / "stage2_results.csv"
BEST_JSON_PATH = RESULTS_DIR / "stage2_best.json"

# Stage 1 best params per dataset (from lgn-s1-best_valid_per_dataset.csv)
STAGE1_BEST = {
    "Amazon-Book":    {"begin_adv": 30, "ema": 1.0, "num_codebook": 512},
    "MIND":           {"begin_adv": 15, "ema": 1.0, "num_codebook": 512},
    "Toys_and_Games": {"begin_adv": 15, "ema": 0.0, "num_codebook": 512},
    "Yelp":           {"begin_adv": 30, "ema": 1.0, "num_codebook": 256},
}

# LightGCN lr / weight_decay per dataset
LR_WD = {
    "Yelp":           {"lr": 1e-2, "weight_decay": 1e-4},
    "MIND":           {"lr": 1e-2, "weight_decay": 1e-4},
    "Amazon-Book":    {"lr": 1e-2, "weight_decay": 1e-5},
    "Toys_and_Games": {"lr": 1e-3, "weight_decay": 1e-4},
}

# Fixed hyperparameters
FIXED = {
    "method":            "PRIDE",
    "weight_mode":       "lambda_power",
    "ablation":          "full",
    "beta":              0.1,
    "drop_rate":         0.1,
    "num_gradual":       10000,
    "n_epochs":          100,
    "patience":          100,
    "batch_size":        2048,
    "test_batch_size":   2048,
    "min_epochs":        0,
    "num_hirearchy":     1,
    "seed":              2024,
    "min_interaction":   10,
    "noise":             0.0,
    "add_p":             1.0,
    "out_dim":           64,
    "val_interval":      1,
    "temp":              0,
    "item_num":          0,
    "alpha":             0,
    "gamma":             0,
    "relabel_ratio":     0,
    "co_lambda":         0,
    "mean_loss_interval":0,
    "energy_gamma":      1.0,
    "lambda_dis":        1.0,
    "tau":               1.0,
    "weight_eps":        1e-8,
    "wgm_alpha":         0.5,
    "lambda_mix":        0.5,
    "gate_tau":          1.0,
}

# Sweep space
MODEL     = "LightGCN"
DATASETS  = ["Yelp", "MIND", "Amazon-Book", "Toys_and_Games"]
ENERGY_R  = [1, 2, 4, 6]
ENERGY_LAMBDA = [0.25, 0.5, 0.75]

CSV_HEADER = [
    "run_id", "dataset", "energy_r", "energy_lambda",
    "begin_adv", "ema", "num_codebook",
    "test_Recall@20", "test_Recall@50", "test_NDCG@20", "test_NDCG@50",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_id(dataset, energy_r, energy_lam):
    return f"{MODEL}_{dataset}_r{energy_r}_l{energy_lam}"


def done_marker(rid):
    return DONE_DIR / f"{rid}.done"


def log_path(rid):
    return LOG_DIR / f"{rid}.log"


def build_cmd(dataset, energy_r, energy_lam, device_id):
    best  = STAGE1_BEST[dataset]
    lr_wd = LR_WD[dataset]
    cmd = [
        sys.executable, str(PROJECT_ROOT / "main.py"),
        "--model",         MODEL,
        "--dataset",       dataset,
        "--device_id",     str(device_id),
        "--energy_r",      str(energy_r),
        "--energy_lambda", str(energy_lam),
        "--begin_adv",     str(best["begin_adv"]),
        "--ema",           str(best["ema"]),
        "--num_codebook",  str(best["num_codebook"]),
        "--lr",            str(lr_wd["lr"]),
        "--weight_decay",  str(lr_wd["weight_decay"]),
    ]
    for k, v in FIXED.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def parse_test_metrics(log_text):
    """Extract final test metrics from captured stdout."""
    test_pos = log_text.rfind("Test - Time:")
    section  = log_text[test_pos:] if test_pos != -1 else log_text

    def last_val(pattern):
        matches = re.findall(pattern, section)
        return float(matches[-1]) if matches else None

    return {
        "test_Recall@20": last_val(r"Recall@20:\s*([\d.]+)"),
        "test_Recall@50": last_val(r"Recall@50:\s*([\d.]+)"),
        "test_NDCG@20":   last_val(r"NDCG@20:\s*([\d.]+)"),
        "test_NDCG@50":   last_val(r"NDCG@50:\s*([\d.]+)"),
    }


def append_csv(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_best_json(all_results):
    """Find best energy_r / energy_lambda per dataset by test_Recall@20."""
    best = {}
    for row in all_results:
        key = row["dataset"]
        if row["test_Recall@20"] is None:
            continue
        if key not in best or row["test_Recall@20"] > best[key]["test_Recall@20"]:
            best[key] = row

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_JSON_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n[Stage2] Best params saved → {BEST_JSON_PATH}")


# ---------------------------------------------------------------------------
# Per-dataset parallel runner
# ---------------------------------------------------------------------------

def run_dataset(dataset, gpu_ids, dry_run):
    """Run all 12 energy_r × energy_lambda combinations for one dataset.

    Returns list of result dicts collected during this dataset's sweep.
    """
    configs = list(product(ENERGY_R, ENERGY_LAMBDA))
    total   = len(configs)
    print(f"\n{'='*60}")
    print(f"[Stage2] Dataset: {dataset}  ({total} runs, GPUs={gpu_ids})")
    print(f"{'='*60}")

    pending = []
    skipped = 0
    for er, el in configs:
        rid = run_id(dataset, er, el)
        if done_marker(rid).exists():
            skipped += 1
        else:
            pending.append((er, el, rid))

    if skipped:
        print(f"[Stage2] Skipping {skipped} already-completed runs.")
    print(f"[Stage2] Pending: {len(pending)}")

    if dry_run:
        for er, el, rid in pending:
            cmd = build_cmd(dataset, er, el, gpu_ids[0])
            print(f"  DRY-RUN [{rid}]: {' '.join(cmd)}")
        return []

    gpu_pool       = list(gpu_ids)
    running        = {}   # rid → (proc, gpu_id, meta, log_file)
    pending_iter   = iter(pending)
    results        = []

    def launch_next(gid):
        try:
            er, el, rid = next(pending_iter)
        except StopIteration:
            return False
        cmd = build_cmd(dataset, er, el, gid)
        lf  = open(log_path(rid), "w")
        print(f"[{datetime.now():%H:%M:%S}] START  {rid}  GPU={gid}")
        proc = subprocess.Popen(
            cmd,
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
        running[rid] = (proc, gid, {"dataset": dataset, "energy_r": er, "energy_lambda": el}, lf)
        return True

    # Fill GPU slots initially
    for gid in list(gpu_pool):
        if not launch_next(gid):
            break

    while running:
        time.sleep(5)
        for rid in list(running):
            proc, gid, meta, lf = running[rid]
            ret = proc.poll()
            if ret is None:
                continue

            lf.close()
            log_text = log_path(rid).read_text(errors="replace")
            metrics  = parse_test_metrics(log_text)
            status   = "OK" if ret == 0 else f"FAILED(rc={ret})"
            print(f"[{datetime.now():%H:%M:%S}] DONE   {rid}  GPU={gid}  {status}  "
                  f"Recall@20={metrics['test_Recall@20']}")

            row = {
                "run_id": rid,
                **meta,
                **STAGE1_BEST[dataset],
                **metrics,
            }
            append_csv(row)
            results.append(row)

            if ret == 0:
                done_marker(rid).touch()

            del running[rid]
            launch_next(gid)

    print(f"[Stage2] Dataset '{dataset}' complete. ({len(results)} results collected)")
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all(gpu_ids, dry_run):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    total_configs = len(DATASETS) * len(ENERGY_R) * len(ENERGY_LAMBDA)
    print(f"[Stage2] backbone={MODEL}  |  datasets (sequential): {DATASETS}")
    print(f"[Stage2] energy_r={ENERGY_R}  energy_lambda={ENERGY_LAMBDA}")
    print(f"[Stage2] Total configs: {total_configs}  |  GPUs/dataset: {gpu_ids}  |  dry_run={dry_run}")

    all_results = []
    for dataset in DATASETS:
        dataset_results = run_dataset(dataset, gpu_ids, dry_run)
        all_results.extend(dataset_results)

    if not dry_run:
        print("\n[Stage2] All datasets finished.")
        save_best_json(all_results)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 lambda_power sweep (LightGCN, sequential datasets)")
    parser.add_argument(
        "--gpus", type=str, default="0,1,2,3",
        help="Comma-separated GPU IDs to use (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    run_all(gpu_ids, args.dry_run)


if __name__ == "__main__":
    main()
