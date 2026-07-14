#!/bin/bash
# Stage 3 LightGCN — beta sweep via WandB (project: LGN-S3-HP)
#
# Stage 1/2 best params are fixed per dataset; sweeps over beta only.
# Datasets are swept sequentially; within each dataset, 5 SLURM array jobs
# run wandb agents in parallel (one GPU per job, matching 5 beta candidates).
#
# Usage (from project root):
#   bash scripts/run_lgn_s3_wandb.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb

DATASETS=("Yelp" "MIND" "Amazon-Book" "Toys_and_Games")
NUM_JOBS=5
LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

declare -A DS_YAML=(
  [Yelp]="./config/lgn_s3_hp_yelp.yaml"
  [MIND]="./config/lgn_s3_hp_mind.yaml"
  [Amazon-Book]="./config/lgn_s3_hp_amazon_book.yaml"
  [Toys_and_Games]="./config/lgn_s3_hp_toys.yaml"
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
    --job-name="lgn-s3-${DS}" \
    --output="$LOG_DIR/s3_${DS}_%A_%a.out" \
    --error="$LOG_DIR/s3_${DS}_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --wrap="wandb agent $SWEEP_ID"
  )

  echo "Completed: $DS (JobID=$JOBID)"
done

echo "All datasets completed."
