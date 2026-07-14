#!/bin/bash
set -e

export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate requiem

SWEEP_OUTPUT=$(wandb sweep ./config/single_mf_moerq_yelp.yaml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
[[ -n "$SWEEP_ID" ]] || { echo "Failed to parse SWEEP_ID"; echo "$SWEEP_OUTPUT"; exit 1; }
echo "Sweep ID: $SWEEP_ID"

JOBID=$(sbatch --parsable --wait \
  --array=1-1 \
  --job-name="mf-moe-yelp" \
  --output="$LOG_DIR/mf_moe_yelp_%A_%a.out" \
  --error="$LOG_DIR/mf_moe_yelp_%A_%a.err" \
  --gres=gpu:1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=4G \
  --wrap="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate requiem && wandb agent $SWEEP_ID"
)

echo "Done. (JobID=$JOBID)"
