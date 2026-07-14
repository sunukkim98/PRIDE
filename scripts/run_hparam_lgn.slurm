#!/bin/bash
set -e

export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

DATASETS=("Yelp" "MIND" "Amazon-Book" "Toys_and_Games")
BASE_SWEEP="./config/hparam_lgn.yaml"

NUM_JOBS=4
LOG_DIR="./log/slurm"
mkdir -p "$LOG_DIR"

# 실험할 하이퍼파라미터 목록
HPARAMS=("num_codebook" "begin_adv" "ema")

for HP in "${HPARAMS[@]}"; do
  echo "======================================"
  echo "🚀 Hyperparameter Sweep: $HP"
  echo "======================================"

  for DS in "${DATASETS[@]}"; do
    echo "------------------------------"
    echo "📦 Dataset: $DS | HP: $HP"
    echo "------------------------------"

    TMP_SWEEP="/tmp/sweep_${DS}_${HP}.yaml"

    python - <<PY
import yaml

BEST = {
  "Yelp": {
    "lr": 1e-2,
    "weight_decay": 1e-4,
    "begin_adv": 15,
    "beta": 0.2,
    "drop_rate": 0.1,
    "ema": 0.0,
    "num_codebook": 512,
    "num_gradual": 10000,
  },
  "MIND": {
    "lr": 1e-2,
    "weight_decay": 1e-4,
    "begin_adv": 50,
    "beta": 0.2,
    "drop_rate": 0.1,
    "ema": 1.0,
    "num_codebook": 128,
    "num_gradual": 10000,
  },
  "Amazon-Book": {
    "lr": 1e-2,
    "weight_decay": 1e-5,
    "begin_adv": 50,
    "beta": 0.2,
    "drop_rate": 0.1,
    "ema": 1.0,
    "num_codebook": 512,
    "num_gradual": 10000,
  },
  "Toys_and_Games": {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "begin_adv": 15,
    "beta": 0.2,
    "drop_rate": 0.1,
    "ema": 1.0,
    "num_codebook": 512,
    "num_gradual": 10000,
  },
}

SWEEP_VALUES = {
  "num_codebook": [128, 256, 512, 1024],
  "begin_adv": [10, 15, 30, 50],
  "ema": [0.0, 0.3, 0.5, 0.7, 0.99, 1.0],
}

ds = "${DS}"
hp = "${HP}"

cfg = yaml.safe_load(open("${BASE_SWEEP}"))
best = BEST[ds]

# dataset 고정
cfg["parameters"]["dataset"]["values"] = [ds]

# best 값들 전부 고정
cfg["parameters"]["lr"]["values"] = [best["lr"]]
cfg["parameters"]["weight_decay"]["values"] = [best["weight_decay"]]
cfg["parameters"]["begin_adv"]["values"] = [best["begin_adv"]]
cfg["parameters"]["beta"]["values"] = [best["beta"]]
cfg["parameters"]["drop_rate"]["values"] = [best["drop_rate"]]
cfg["parameters"]["ema"]["values"] = [best["ema"]]
cfg["parameters"]["num_codebook"]["values"] = [best["num_codebook"]]
cfg["parameters"]["num_gradual"]["values"] = [best["num_gradual"]]

# sweep 대상 파라미터 하나만 values 여러 개로 열기
cfg["parameters"][hp]["values"] = SWEEP_VALUES[hp]

yaml.safe_dump(cfg, open("${TMP_SWEEP}", "w"), sort_keys=False)
print(f"Wrote: ${TMP_SWEEP}")
print(f"Dataset={ds}, HParam={hp}, Sweep={SWEEP_VALUES[hp]}")
PY

    SWEEP_OUTPUT=$(wandb sweep "$TMP_SWEEP" 2>&1)
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
    [[ -n "$SWEEP_ID" ]] || { echo "❌ Failed to parse SWEEP_ID"; echo "$SWEEP_OUTPUT"; exit 1; }
    echo "✅ Sweep ID: $SWEEP_ID"

    echo "🧩 Submitting & waiting until completion: $DS | $HP (agents=$NUM_JOBS)"
    JOBID=$(sbatch --parsable --wait \
      --exclude=node5 \
      --array=1-"$NUM_JOBS" \
      --job-name="mf-${HP}-${DS}" \
      --output="$LOG_DIR/${HP}_${DS}_%A_%a.out" \
      --error="$LOG_DIR/${HP}_${DS}_%A_%a.err" \
      --gres=gpu:1 \
      --ntasks=1 \
      --cpus-per-task=1 \
      --mem=4G \
      --wrap="wandb agent $SWEEP_ID"
    )

    echo "✅ Completed dataset: $DS | $HP (JobID=$JOBID)"
  done
done

echo "🎉 All hyperparameter sweeps completed sequentially."