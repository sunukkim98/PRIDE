#!/bin/bash
set -e

export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

DATASETS=("Gowalla" "Yelp" "MIND" "Amazon-Book" "Toys_and_Games" "Baby")
BASE_SWEEP="./config/sweep_neumf_origin.yaml"

NUM_JOBS=5
LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

for DS in "${DATASETS[@]}"; do
  echo "=============================="
  echo "🚀 Dataset: $DS"
  echo "=============================="

  TMP_SWEEP="/tmp/sweep_${DS}.yaml"

  python - <<PY
import yaml
cfg = yaml.safe_load(open("${BASE_SWEEP}"))
cfg["parameters"]["dataset"]["values"] = ["${DS}"]
yaml.safe_dump(cfg, open("${TMP_SWEEP}", "w"), sort_keys=False)
print("Wrote:", "${TMP_SWEEP}")
PY

  SWEEP_OUTPUT=$(wandb sweep "$TMP_SWEEP" 2>&1)
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
  [[ -n "$SWEEP_ID" ]] || { echo "❌ Failed to parse SWEEP_ID"; echo "$SWEEP_OUTPUT"; exit 1; }
  echo "✅ Sweep ID: $SWEEP_ID"

  echo "🧩 Submitting & waiting until completion: $DS (agents=$NUM_JOBS)"
  JOBID=$(sbatch --parsable --wait \
    --array=1-"$NUM_JOBS" \
    --job-name="lgn-origin-${DS}-hp" \
    --output="$LOG_DIR/${DS}_%A_%a.out" \
    --error="$LOG_DIR/${DS}_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --wrap="wandb agent $SWEEP_ID"
  )

  echo "✅ Completed dataset: $DS (JobID=$JOBID)"
done

echo "🎉 All datasets completed sequentially."
