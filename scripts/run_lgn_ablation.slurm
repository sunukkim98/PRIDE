#!/bin/bash
set -e

export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

DATASETS=("Yelp" "MIND" "Amazon-Book" "Toys_and_Games")
BASE_SWEEP="./config/ablation_moe_rq_lgn.yaml"

NUM_JOBS=1
LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

for DS in "${DATASETS[@]}"; do
  echo "=============================="
  echo "🚀 Dataset: $DS"
  echo "=============================="

  TMP_SWEEP="/tmp/sweep_${DS}.yaml"

  python - <<PY
import yaml

BEST = {
  "Gowalla": (1e-3, 1e-3),
  "Yelp": (1e-2, 1e-4, 15, 0.2, 0.1, 0.0, 512, 10000),
  "MIND": (1e-2, 1e-4, 50, 0.2, 0.1, 1.0, 128, 10000),
  "Amazon-Book": (1e-2, 1e-5, 50, 0.2, 0.1, 1.0, 512, 1000),
  "Baby": (1e-4, 1e-1),
  "Toys_and_Games": (1e-3, 1e-4, 15, 0.2, 0.1, 1.0, 512, 10000),
}

ds = "${DS}"
lr, wd, begin, beta, drop_rate, ema, codebook, num_gradual = BEST[ds]

cfg = yaml.safe_load(open("${BASE_SWEEP}"))

# dataset
cfg["parameters"]["dataset"]["values"] = [ds]

# lr/wd를 "고정(value)"로 박는 걸 추천 (불필요한 grid 방지)
cfg["parameters"]["lr"] = {"values": [float(lr)]}
cfg["parameters"]["weight_decay"] = {"values": [float(wd)]}
cfg["parameters"]["begin_adv"] = {"values": [float(begin)]}
cfg["parameters"]["beta"] = {"values": [float(beta)]}
cfg["parameters"]["drop_rate"] = {"values": [float(drop_rate)]}
cfg["parameters"]["ema"] = {"values": [float(ema)]}
cfg["parameters"]["num_codebook"] = {"values": [int(codebook)]}
cfg["parameters"]["num_gradual"] = {"values": [int(num_gradual)]}

yaml.safe_dump(cfg, open("${TMP_SWEEP}", "w"), sort_keys=False)
print("Wrote:", "${TMP_SWEEP}", "with lr/wd=", lr, wd)
PY

  SWEEP_OUTPUT=$(wandb sweep "$TMP_SWEEP" 2>&1)
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
  [[ -n "$SWEEP_ID" ]] || { echo "❌ Failed to parse SWEEP_ID"; echo "$SWEEP_OUTPUT"; exit 1; }
  echo "✅ Sweep ID: $SWEEP_ID"

  echo "🧩 Submitting & waiting until completion: $DS (agents=$NUM_JOBS)"
  JOBID=$(sbatch --parsable --wait \
    --array=1-"$NUM_JOBS" \
    --job-name="lgn-ablation-${DS}" \
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
