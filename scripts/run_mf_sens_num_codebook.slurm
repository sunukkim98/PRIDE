#!/bin/bash
# MF backbone — num_codebook sensitivity sweep
# Datasets: Amazon-Book, MIND, Yelp  (5 values each → array 1-5)
#
# Usage:
#   bash scripts/run_mf_sens_num_codebook.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb ./log/slurm

HP="num_codebook"
NUM_JOBS=5   # [64, 128, 256, 512, 1024]
LOG_DIR="./log/slurm"

DATASETS=("Amazon-Book" "MIND" "Yelp")
declare -A DS_SLUG=(
  [Amazon-Book]="amazon_book"
  [MIND]="mind"
  [Yelp]="yelp"
)

echo "======================================"
echo "HP sweep: $HP  ($NUM_JOBS values)"
echo "Datasets: ${DATASETS[*]}"
echo "======================================"

for DS in "${DATASETS[@]}"; do
  SLUG="${DS_SLUG[$DS]}"
  YAML="./config/sensitivity/mf_sens_${HP}_${SLUG}.yaml"

  if [[ ! -f "$YAML" ]]; then
    echo "  [SKIP] $YAML not found"
    continue
  fi

  echo "  Dataset: $DS"

  SWEEP_OUTPUT=$(wandb sweep "$YAML" 2>&1)
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
  if [[ -z "$SWEEP_ID" ]]; then
    echo "  [ERROR] Failed to parse SWEEP_ID"
    echo "$SWEEP_OUTPUT"
    exit 1
  fi
  echo "  Sweep ID: $SWEEP_ID"

  JOBID=$(sbatch --parsable --wait \
    --exclude=node5 \
    --array=1-"${NUM_JOBS}" \
    --job-name="mf-sens-${HP}-${SLUG}" \
    --output="${LOG_DIR}/sens_${HP}_${SLUG}_%A_%a.out" \
    --error="${LOG_DIR}/sens_${HP}_${SLUG}_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --wrap="wandb agent --count 1 $SWEEP_ID || true"
  )
  echo "  Done: $DS | $HP  (JobID=$JOBID)"
done

echo ""
echo "num_codebook sensitivity sweep completed."
