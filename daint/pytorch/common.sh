# Shared training setup for run.sh and run_stat.sh.
: "${SLURM_JOB_ID:?Submit with sbatch or run inside an salloc allocation}"
: "${SLURM_JOB_NUM_NODES:?Slurm did not provide the allocation size}"
: "${SLURM_JOB_NODELIST:?Slurm did not provide the node list}"
: "${SCRATCH:?SCRATCH must be set}"

# Use the first allocated node and a job-specific port for rendezvous.
rdzv_host=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')

: "${rdzv_host:?Could not determine the first allocated node}"
rdzv_endpoint="$rdzv_host:${TORCH_RDZV_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"

# Keep logs, checkpoints, and stack graphs together on scratch.
export TRAIN_OUTPUT_DIR="$SCRATCH/pytorch-example/runs/$SLURM_JOB_ID-$(date -u +%Y%m%d-%H%M%S)-$$"
mkdir -p "$TRAIN_OUTPUT_DIR"
cd "$TRAIN_OUTPUT_DIR"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ulimit -c 0

# Slurm starts one torchrun per node, and torchrun starts four GPU workers.
training_command=(
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1
    --cpu-bind=none --mpi=pmix --environment="$PYTORCH_EXAMPLE_DIR/pytorch.toml"

    torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=4
    --rdzv-backend=c10d --rdzv-endpoint="$rdzv_endpoint" --rdzv-id="$SLURM_JOB_ID"

    /workspace/train.py --steps "${TRAIN_STEPS:-1000000}" "$@"
)

printf 'Job=%s nodes=%s rendezvous=%s\nResults=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$rdzv_endpoint" "$TRAIN_OUTPUT_DIR"
