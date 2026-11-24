#!/usr/bin/env bash
#SBATCH --account=gxx-0
#SBATCH --partition=debug
#SBATCH --job-name=pytorch-stat
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

# Use the shared STAT installation on Daint.
export STAT_PREFIX=/ritom/scratch/cscs/kumbhar/software/stat-home/install
STAT_MODE=${STAT_MODE:-attach}

# STAT takes the delay in seconds and the sample interval in milliseconds.
stat_command=(
    "$STAT_PREFIX/bin/stat-cl"
    --backend pystack
    -i
    -S
    --sleep "${STAT_DELAY:-120}"
    -t "${STAT_SAMPLES:-1}"
    -T "$((1000 * ${STAT_INTERVAL:-5}))"
    -j "$SLURM_JOB_ID-$STAT_MODE"
)

# children and native stacks
[[ ${STAT_CHILDREN:-1} == 1 ]] && stat_command+=(--with-children)
[[ ${STAT_NATIVE:-0} == 1 ]] && stat_command+=(--native)

# attach
case "$STAT_MODE" in

    launch)
        # Let STAT launch srun and collect the samples.
        time "${stat_command[@]}" -C "${training_command[@]}"
        ;;

    attach)
        # Start srun, sample by launcher PID, then wait for training.
        "${training_command[@]}" > training.log 2>&1 &
        launcher_pid=$!
        printf 'srun PID=%s\nTraining log=%s/training.log\n' "$launcher_pid" "$TRAIN_OUTPUT_DIR"

        stat_status=0
        time "${stat_command[@]}" "$launcher_pid" || stat_status=$?
        wait "$launcher_pid"
        exit "$stat_status"
        ;;

    *)
        echo 'STAT_MODE must be attach or launch.' >&2
        exit 2
        ;;
esac
