#!/bin/bash
set -e  # Exit script immediately on error
export WANDB_DIR=./log
export LOG_DISABLE=1
export TQDM_DISABLE=1

# Default values
SWEEP_CONFIG="./config/ablation_moe_rq_lgn.yaml"
NUM_JOBS=4
LOG_DIR="./log/slurm"

# Function to display usage instructions
usage() {
    echo "Usage: $0 --config=path/to/sweep.yaml --num_jobs=N"
    echo "  --config=path     Path to the WandB sweep configuration file"
    echo "  --num_jobs=N      Number of SLURM jobs to launch"
    echo "  --help            Show this help message"
    exit 1
}

# Parse command-line arguments using getopt
OPTIONS=$(getopt -o "" --long config:,num_jobs:,help -- "$@") || usage
eval set -- "$OPTIONS"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) SWEEP_CONFIG="$2"; shift 2 ;;  # Set sweep configuration file
        --num_jobs) NUM_JOBS="$2"; shift 2 ;;  # Set the number of SLURM jobs
        --help) usage ;;  # Display help message and exit
        --) shift; break ;;  # End of options
        *) usage ;;  # Invalid option
    esac
done

# Validate sweep configuration file existence
[[ -f "$SWEEP_CONFIG" ]] || {
  echo "❌ Error: Sweep configuration file '$SWEEP_CONFIG' not found."; usage;
}

# Validate that NUM_JOBS is a positive integer
[[ "$NUM_JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "❌ Error: Number of jobs must be a positive integer."; usage;
}

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Create a WandB sweep and extract the Sweep ID
echo "🚀 Creating a WandB sweep with config: $SWEEP_CONFIG ..."
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1) || {
    echo "❌ Error: WandB sweep creation failed!"
    echo "📜 WandB error log:"
    echo "$SWEEP_OUTPUT"
    exit 1
}

# Extract the Sweep ID using awk for better reliability
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent/ {print $NF}')
if [[ -z "$SWEEP_ID" ]]; then
  echo "❌ Failed to retrieve Sweep ID."
  echo "📜 WandB output log:"
  echo "$SWEEP_OUTPUT"
  exit 1
fi

echo "✅ WandB Sweep created successfully: $SWEEP_ID"

# Submit SLURM jobs
echo "📡 Submitting SLURM jobs ($NUM_JOBS WandB agents in total)..."
sbatch --array=1-"$NUM_JOBS" --job-name=mf-ablation \
    --output="$LOG_DIR/agent_%A_%a.out" \
    --error="$LOG_DIR/agent_%A_%a.err" \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=1G \
    --wrap="wandb agent $SWEEP_ID"

echo "✅ SLURM job submission complete: $NUM_JOBS WandB agents running."