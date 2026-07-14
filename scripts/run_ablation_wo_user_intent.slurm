#!/bin/bash
# Ablation: w/o User Intent Weight — single run per dataset (MF backbone)
# Full model best params 고정, ablation='wo_user_intent'
#
# Usage (from project root):
#   bash scripts/run_ablation_wo_user_intent.slurm

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
  [Yelp]="./config/ablation/wo_user_intent_yelp.yaml"
  [MIND]="./config/ablation/wo_user_intent_mind.yaml"
  [Amazon-Book]="./config/ablation/wo_user_intent_amazon_book.yaml"
  [Toys_and_Games]="./config/ablation/wo_user_intent_toys.yaml"
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
    --job-name="lgn-abl-wui-${DS}" \
    --output="$LOG_DIR/abl_wui_${DS}_%A_%a.out" \
    --error="$LOG_DIR/abl_wui_${DS}_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --wrap="wandb agent $SWEEP_ID"
  )

  echo "Completed: $DS (JobID=$JOBID)"
done

echo "All datasets completed (w/o User Intent Weight)."
