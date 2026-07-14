#!/bin/bash
#SBATCH --job-name=mf-moe-mind
#SBATCH --output=./log/slurm/mf_moe_mind_%j.out
#SBATCH --error=./log/slurm/mf_moe_mind_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

mkdir -p ./log/slurm

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate requiem

python main.py \
  --model            MF \
  --dataset          MIND \
  --method           PRIDE \
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
