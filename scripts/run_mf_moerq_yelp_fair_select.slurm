#!/bin/bash
set -e
export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

SWEEP_CONFIG="./config/mf_moerq_yelp_fair_select.yaml"
NUM_JOBS=1
LOG_DIR="./log/slurm"

mkdir -p "$LOG_DIR"

echo "======================================"
echo "MF / Yelp / PRIDE (FairSelect)"
echo "======================================"

echo "Creating WandB sweep: $SWEEP_CONFIG ..."
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1) || {
    echo "Error: WandB sweep creation failed!"
    echo "$SWEEP_OUTPUT"
    exit 1
}

SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
if [[ -z "$SWEEP_ID" ]]; then
    echo "Failed to retrieve Sweep ID."
    echo "$SWEEP_OUTPUT"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"

echo "Submitting SLURM job ..."
sbatch --array=1-"$NUM_JOBS" \
    --job-name="mf-moerq-fair-yelp" \
    --output="$LOG_DIR/mf_moerq_fair_yelp_%A_%a.out" \
    --error="$LOG_DIR/mf_moerq_fair_yelp_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --wrap="wandb agent $SWEEP_ID"

echo "Done."
