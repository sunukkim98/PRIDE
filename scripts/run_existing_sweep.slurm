#!/bin/bash
#SBATCH --job-name=wandb-agent
#SBATCH --output=./log/slurm/agent_%A_%a.out
#SBATCH --error=./log/slurm/agent_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=1-4

export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

# 기존 sweep ID를 여기에 입력하세요
SWEEP_ID="forward-rec/requiem/c6zzzxra"

# 현재 SLURM 작업의 GPU 번호를 환경변수로 설정 (선택적, 코드에서 device_id 사용시)
export CUDA_VISIBLE_DEVICES=0

wandb agent $SWEEP_ID