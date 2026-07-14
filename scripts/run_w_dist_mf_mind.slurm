#!/bin/bash
# w distribution analysis — MF / MIND
#
# noise in [0.0, 0.1, 0.2, 0.3, 0.4] (5 runs) with MIND best hyperparameters.
# 5 array jobs run in parallel (one per noise level).
# After all jobs finish, run the plot script automatically.
#
# Usage (from project root):
#   bash scripts/run_w_dist_mf_mind.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb

YAML="./config/w_dist_mf_mind.yaml"
NUM_JOBS=5   # noise 5개를 병렬로
LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Dataset: MIND  |  noise: 0.0~0.4"
echo "======================================"

SWEEP_OUTPUT=$(wandb sweep "$YAML" 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
[[ -n "$SWEEP_ID" ]] || { echo "Failed to parse SWEEP_ID"; echo "$SWEEP_OUTPUT"; exit 1; }
echo "Sweep ID: $SWEEP_ID"

echo "Submitting array jobs (agents=$NUM_JOBS, runs=5) ..."
JOBID=$(sbatch --parsable --wait \
  --nodelist=node3 \
  --array=1-"$NUM_JOBS"%4 \
  --job-name="w-dist-mf-MIND" \
  --output="$LOG_DIR/w_dist_mf_MIND_%A_%a.out" \
  --error="$LOG_DIR/w_dist_mf_MIND_%A_%a.err" \
  --gres=gpu:1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=4G \
  --wrap="wandb agent $SWEEP_ID"
)

echo "Training completed (JobID=$JOBID)"

echo "======================================"
echo "Plotting w distributions ..."
echo "======================================"
conda run -n requiem python analyze/plot_w_dist.py \
  --dataset MIND \
  --noise_ratios 0.0 0.1 0.2 0.3 0.4 \
  --seeds 2024 \
  --n_epochs_show 4 \
  --out_dir analyze/figures

echo "Done. Figures saved to analyze/figures/w_dist_MIND.{pdf,png}"

echo "======================================"
echo "Plotting REQUIEM signal distributions ..."
echo "======================================"
conda run -n requiem python analyze/plot_requiem_signals.py \
  --dataset MIND \
  --noise_ratios 0.0 0.1 0.2 0.3 0.4 \
  --seeds 2024 \
  --signal w \
  --n_epochs_show 4 \
  --out_dir analyze/figures

echo "Done. Figures saved to analyze/figures/requiem_*_MIND.{pdf,png}"
