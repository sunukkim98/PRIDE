#!/bin/bash
# MF backbone RISE — hyperparameter sensitivity sweep (project: MF-Sensitivity)
#
# Each HP is swept independently (one at a time) while all others are fixed
# at the dataset-specific best values.
#
# Execution order:  HP (outer) → dataset (inner) — sequential with --wait.
# Parallel agents per sweep = number of sweep values for that HP.
#
# Usage (from project root):
#   bash scripts/run_mf_sensitivity.slurm
#
# Optional: run a single HP only:
#   HP_FILTER=ema bash scripts/run_mf_sensitivity.slurm

set -e

export WANDB_DIR=./log/wandb
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/wandb ./log/slurm

DATASETS=("Amazon-Book" "MIND" "Toys_and_Games" "Yelp")

# file-name slugs (must match config/sensitivity/mf_sens_{HP}_{SLUG}.yaml)
declare -A DS_SLUG=(
  [Amazon-Book]="amazon_book"
  [MIND]="mind"
  [Toys_and_Games]="toys"
  [Yelp]="yelp"
)

# number of sweep values per HP  →  number of parallel wandb agents
declare -A NUM_JOBS=(
  [begin_adv]=4
  [ema]=6
  [num_codebook]=3
  [energy_r]=4
  [energy_lambda]=3
  [beta]=5
  [drop_rate]=3
  [num_gradual]=5
)

HPS=(begin_adv ema num_codebook energy_r energy_lambda beta drop_rate num_gradual)

LOG_DIR="./log/slurm"

for HP in "${HPS[@]}"; do
  # allow filtering to a single HP: HP_FILTER=ema bash ...
  [[ -n "${HP_FILTER:-}" && "$HP" != "$HP_FILTER" ]] && continue

  echo "======================================"
  echo "HP sweep: $HP  (${NUM_JOBS[$HP]} values)"
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
      --array=1-"${NUM_JOBS[$HP]}" \
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
done

echo ""
echo "All sensitivity sweeps completed."
