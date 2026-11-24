#!/usr/bin/env bash
#SBATCH --account=gxx-0
#SBATCH --partition=debug
#SBATCH --job-name=pytorch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Slurm runs a spooled copy of this script. Use the submission directory.
export PYTORCH_EXAMPLE_DIR=${PYTORCH_EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$PYTORCH_EXAMPLE_DIR/common.sh"

# Run the shared training command in the foreground.
exec "${training_command[@]}"
