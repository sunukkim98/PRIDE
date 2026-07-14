#!/usr/bin/env bash
# Single-run script: MF / MIND / PRIDE (validated hyperparameters)
# Usage: bash scripts/run_pride_mf_mind.sh [GPU_ID]

GPU_ID=${1:-0}

cd "$(dirname "$0")/.."

python main.py \
  --model            MF \
  --dataset          MIND \
  --method           PRIDE \
  --device_id        "$GPU_ID" \
  --seed             2024 \
  --n_epochs         100 \
  --patience         100 \
  --min_epochs       0 \
  --val_interval     1 \
  --batch_size       2048 \
  --test_batch_size  2048 \
  --out_dim          64 \
  --lr               0.001 \
  --weight_decay     0.001 \
  --min_interaction  10 \
  --noise            0 \
  --add_p            1 \
  --begin_adv        15 \
  --ema              0.75 \
  --num_codebook     512 \
  --num_hirearchy    1 \
  --weight_mode      lambda_power \
  --energy_r         4 \
  --energy_lambda    0.5 \
  --energy_gamma     1 \
  --lambda_dis       1 \
  --tau              1 \
  --weight_eps       0.00000001 \
  --wgm_alpha        0.5 \
  --lambda_mix       0.5 \
  --ablation         full \
  --beta             0.1 \
  --drop_rate        0.2 \
  --num_gradual      30000 \
  --gate_tau         1 \
  --alpha            0 \
  --gamma            0 \
  --relabel_ratio    0 \
  --co_lambda        0 \
  --mean_loss_interval 0 \
  --temp             0 \
  --item_num         0
