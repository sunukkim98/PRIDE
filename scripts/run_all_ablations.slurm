#!/bin/bash
# Run all 3 ablation variants sequentially (MF backbone)
# Total: 12 runs, ~12 hours
#
# Usage (from project root):
#   bash scripts/run_all_ablations.slurm
#
# To run variants in parallel instead:
#   bash scripts/run_ablation_wo_user_intent.slurm &
#   bash scripts/run_ablation_wo_stability.slurm &
#   bash scripts/run_ablation_wo_requiem.slurm &
#   wait

set -e

echo "Starting all ablation experiments (MF backbone)..."

echo "=== Ablation 1/3: w/o User Intent Weight ==="
bash scripts/run_ablation_wo_user_intent.slurm

echo "=== Ablation 2/3: w/o Stability Weight ==="
bash scripts/run_ablation_wo_stability.slurm

echo "=== Ablation 3/3: w/o both weights (wo_requiem) ==="
bash scripts/run_ablation_wo_requiem.slurm

echo "All ablation experiments completed."
