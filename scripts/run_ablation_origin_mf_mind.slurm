#!/bin/bash
# Ablation baseline: MF Origin / MIND
# lr=0.001, weight_decay=0.001 (same as expert_only_mf_mind)
#
# Usage (from project root):
#   bash scripts/run_ablation_origin_mf_mind.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb
mkdir -p ./log/slurm

LOG_DIR="./log/slurm"
YAML="./config/ablation/origin_mf_mind.yaml"

echo "======================================"
echo "  MF Origin / MIND"
echo "======================================"

SWEEP_OUTPUT=$(wandb sweep "$YAML" 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
[[ -n "$SWEEP_ID" ]] || { echo "Failed to parse SWEEP_ID"; echo "$SWEEP_OUTPUT"; exit 1; }
echo "Sweep ID: $SWEEP_ID"

JOBID=$(sbatch --parsable \
  --exclude=node5 \
  --array=1-1 \
  --job-name="mf-origin-mind" \
  --output="$LOG_DIR/mf_origin_mind_%A_%a.out" \
  --error="$LOG_DIR/mf_origin_mind_%A_%a.err" \
  --gres=gpu:1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=4G \
  --wrap="wandb agent $SWEEP_ID"
)

echo "Completed: MF Origin / MIND (JobID=$JOBID)"
