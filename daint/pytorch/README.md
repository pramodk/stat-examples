# PyTorch and STAT on Daint

- `run.sh` starts plain PyTorch training for attaching from the desktop GUI with STAT Remote Connect.
- `run_stat.sh` demonstrates command-line sampling with `stat-cl`.

Both scripts use `common.sh` to start one `torchrun` per node and four GPU workers per node. The default
allocation has two nodes, so the training workers have global ranks 0–7.

## Set up on Daint

Connect to Daint and check out this repository there. From the root of that checkout, change into the example
directory before submitting either script.

```console
cd daint/pytorch
```

Both scripts request two nodes, four GPUs per node, and ten minutes in the `debug` partition. The scripts
specify account `gxx-0`. Pass `--account=YOUR_ACCOUNT` to `sbatch` if you use a different account.

The Container Engine uses NVIDIA PyTorch `26.09-py3` from `pytorch.toml`. The first run may take longer while
the image is pulled and cached.

## Attach from the desktop GUI

Submit the training job on Daint.

```console
sbatch run.sh
```

Use `sbatch --nodes=1 run.sh` for a one-node run, or set `TRAIN_STEPS` to shorten the workload. Both scripts
default to 1,000,000 steps, subject to the Slurm time limit.

Once training starts, open STAT Remote Connect from your desktop. Replace the prefix with the path to STAT
on Daint, then select the job ID from `sbatch` and attach to its `srun` launcher.

```console
stat-gui --remote-login daint --remote-prefix /path/to/stat-install
```

## Sample with the STAT CLI

`run_stat.sh` has a `STAT_PREFIX` setting near the top. If that installation is unavailable to you, set it in
the script to the location of your STAT installation before submitting.

```console
sbatch run_stat.sh
```

By default, the script starts training, waits 120 seconds before attaching to the `srun` launcher, and
collects one PyStack sample with child processes included. Increase `STAT_DELAY` if the container needs more
time to start. Leave enough time for STAT to attach, sample, and detach before the job ends.

Collect three samples four seconds apart, including Python and native C/C++ frames.

```console
STAT_NATIVE=1 STAT_SAMPLES=3 STAT_INTERVAL=4 sbatch run_stat.sh
```

To have STAT start the training step itself, use launch mode.

```console
STAT_MODE=launch sbatch run_stat.sh
```

| Setting | Default | What it controls |
| --- | --- | --- |
| `STAT_DELAY` | `120` | Seconds to wait before attach in attach mode, or before sampling in launch mode. |
| `STAT_SAMPLES` | `1` | Number of stack samples. |
| `STAT_INTERVAL` | `5` | Seconds between samples. |
| `STAT_CHILDREN` | `1` | Set to `0` to exclude child processes. |
| `STAT_NATIVE` | `0` | Set to `1` to include native C/C++ frames. |

## Watch the job and read results

```console
squeue --me
tail -f slurm-pytorch-stat-JOBID.out
```

Replace `JOBID` with the ID from `sbatch`. For plain training, follow `slurm-pytorch-JOBID.out`.

The Slurm log prints a `Results=` path under `$SCRATCH/pytorch-example/runs/`. Look there for the latest
checkpoint, `checkpoint.pt`, once training writes it. In attach mode, the same directory contains
`training.log` for live training output. STAT prints the path to its graphs under `stat_results/`.

Run `stat-dot-tree` from the `bin` directory of your STAT installation to print a graph as a text tree.

```console
/ritom/scratch/cscs/kumbhar/software/stat-home/install/bin/stat-dot-tree /path/to/sample.2D.dot
```

A `2D.dot` file represents one sample. A `3D.dot` file combines multiple samples.
