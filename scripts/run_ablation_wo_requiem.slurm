#!/bin/bash
# Ablation: w/o both weights — single run per dataset (MF backbone)
# Full model best params 고정, ablation='wo_requiem'
# NOTE: energy_r / energy_lambda are passed but ignored (weight=1 fixed internally)
#
# Usage (from project root):
#   bash scripts/run_ablation_wo_requiem.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb

DATASETS=("Yelp" "MIND" "Amazon-Book" "Toys_and_Games")
NUM_JOBS=1
LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

declare -A DS_YAML=(
  [Yelp]="./config/ablation/wo_requiem_yelp.yaml"
  [MIND]="./config/ablation/wo_requiem_mind.yaml"
  [Amazon-Book]="./config/ablation/wo_requiem_amazon_book.yaml"
  [Toys_and_Games]="./config/ablation/wo_requiem_toys.yaml"
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

  echo "Submitting & waiting: $DS (agents=$NUM_JOBS)"
  JOBID=$(sbatch --parsable --wait \
    --exclude=node5 \
    --array=1-"$NUM_JOBS" \
    --job-name="lgn-abl-wrq-${DS}" \
    --output="$LOG_DIR/abl_wrq_${DS}_%A_%a.out" \
    --error="$LOG_DIR/abl_wrq_${DS}_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --wrap="wandb agent $SWEEP_ID"
  )

  echo "Completed: $DS (JobID=$JOBID)"
done

echo "All datasets completed (w/o both weights / wo_requiem)."
