#!/bin/bash
# Ablation: expert selection only (no REQUIEM weighting)
# ablation='wo_requiem': MoE warmup + expert selection, then plain BPR on selected expert
# Runs both MF and LightGCN backbones across all 4 datasets sequentially.
#
# Usage (from project root):
#   bash scripts/run_ablation_expert_only.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb
mkdir -p ./log/slurm

LOG_DIR="./log/slurm"
NUM_JOBS=1

# -------------------------------------------------------
# MF backbone
# -------------------------------------------------------
echo "======================================"
echo "  MF backbone — expert selection only"
echo "======================================"

declare -A MF_YAML=(
  [Yelp]="./config/ablation/expert_only_mf_yelp.yaml"
  [MIND]="./config/ablation/expert_only_mf_mind.yaml"
  [Amazon-Book]="./config/ablation/expert_only_mf_amazon_book.yaml"
  [Toys_and_Games]="./config/ablation/expert_only_mf_toys.yaml"
)

for DS in MIND; do
  echo "--------------------------------------"
  echo "  MF / Dataset: $DS"
  echo "--------------------------------------"

  YAML="${MF_YAML[$DS]}"

  SWEEP_OUTPUT=$(wandb sweep "$YAML" 2>&1)
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
  [[ -n "$SWEEP_ID" ]] || { echo "Failed to parse SWEEP_ID for $DS"; echo "$SWEEP_OUTPUT"; exit 1; }
  echo "Sweep ID: $SWEEP_ID"

  JOBID=$(sbatch --parsable --wait \
    --exclude=node5 \
    --array=1-"$NUM_JOBS" \
    --job-name="mf-exponly-${DS}" \
    --output="$LOG_DIR/mf_exponly_${DS}_%A_%a.out" \
    --error="$LOG_DIR/mf_exponly_${DS}_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --wrap="wandb agent $SWEEP_ID"
  )
  echo "Completed: MF / $DS (JobID=$JOBID)"
done