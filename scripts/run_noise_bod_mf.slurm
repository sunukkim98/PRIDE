#!/bin/bash
# MF BOD noise robustness experiment
#
# Runs noise in [0.0, 0.1, 0.2, 0.3, 0.4] (5 runs) for each of the 4 datasets.
# Best HPs (alpha, gamma) per dataset from mf-bod HP search.
# Datasets are processed sequentially; 5 array jobs run in parallel (one per noise level).
#
# Usage (from project root):
#   bash scripts/run_noise_bod_mf.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb

DATASETS=("Amazon-Book" "MIND" "Toys_and_Games" "Yelp")
NUM_JOBS=5
LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

declare -A DS_YAML=(
  [Amazon-Book]="./config/noise_bod_mf_amazon_book.yaml"
  [MIND]="./config/noise_bod_mf_mind.yaml"
  [Toys_and_Games]="./config/noise_bod_mf_toys.yaml"
  [Yelp]="./config/noise_bod_mf_yelp.yaml"
)

for DS in "${DATASETS[@]}"; do
  echo "======================================"
  echo "Dataset: $DS"
  echo "======================================"

  YAML="${DS_YAML[$DS]}"

  SWEEP_OUTPUT=$(wandb sweep "$YAML" 2>&1)
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
  [[ -n "$SWEEP_ID" ]] || { echo "Failed to parse SWEEP_ID"; echo "$SWEEP_OUTPUT"; exit 1; }
  echo "Sweep ID: $SWEEP_ID"

  echo "Submitting & waiting: $DS (agents=$NUM_JOBS, runs=5)"
  JOBID=$(sbatch --parsable --wait \
    --exclude=node5 \
    --array=1-"$NUM_JOBS" \
    --job-name="mf-bod-noise-${DS}" \
    --output="$LOG_DIR/noise_bod_mf_${DS}_%A_%a.out" \
    --error="$LOG_DIR/noise_bod_mf_${DS}_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --wrap="wandb agent $SWEEP_ID"
  )

  echo "Completed: $DS (JobID=$JOBID)"
done

echo "All datasets completed."
