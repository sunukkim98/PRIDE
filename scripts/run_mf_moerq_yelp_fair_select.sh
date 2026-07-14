#!/usr/bin/env bash
# MF / Yelp / PRIDE (FairSelect) — 로컬 실행
# Usage: bash scripts/run_mf_moerq_yelp_fair_select.sh [GPU_ID]
set -e

GPU_ID=${1:-0}
SWEEP_CONFIG="./config/mf_moerq_yelp_fair_select.yaml"

export TQDM_DISABLE=1
export WANDB_DIR=./log

cd "$(dirname "$0")/.."

echo "======================================"
echo "MF / Yelp / PRIDE (FairSelect)"
echo "GPU: $GPU_ID"
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

CUDA_VISIBLE_DEVICES="$GPU_ID" wandb agent "$SWEEP_ID"

echo "Done."
