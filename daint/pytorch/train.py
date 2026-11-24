"""Synthetic GPU training with DataLoader workers and an async checkpoint writer."""

import argparse
import multiprocessing as mp
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset


def log(message):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] pid={os.getpid()} {message}", flush=True)


class SyntheticData(IterableDataset):
    def __init__(self, width, classes):
        self.width = width
        self.classes = classes

    def __iter__(self):
        while True:
            inputs = torch.randn(self.width)
            target = torch.randint(self.classes, ()).item()
            yield inputs, target


def init_worker(worker_id):
    torch.set_num_threads(1)
    log(f"DataLoader rank={os.environ['RANK']} worker={worker_id} started")


def write_checkpoints(queue, idle, output_dir):
    torch.set_num_threads(1)
    log("Checkpoint writer started")

    while True:
        checkpoint = queue.get()
        if checkpoint is None:
            break

        step, weights = checkpoint
        log(f"Checkpoint write starts: step={step}")

        temporary = output_dir / "checkpoint.tmp"
        destination = output_dir / "checkpoint.pt"
        # Publish the checkpoint after the write completes.
        torch.save({"step": step, "model": weights}, temporary)
        temporary.replace(destination)

        del weights, checkpoint
        log(f"Checkpoint write finished: step={step} path={destination}")
        idle.set()

    log("Checkpoint writer finished")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("TRAIN_OUTPUT_DIR", ".")),
    )
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.width, args.depth, args.checkpoint_every) < 1:
        parser.error("steps, batch size, width, depth and checkpoint interval must be positive")
    if args.workers < 0:
        parser.error("workers must be nonnegative")

    return args


def build_model(width, depth, device):
    # Build the same model on every rank before wrapping it in DDP.
    torch.manual_seed(42)
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(width, width), nn.GELU()])
    layers.append(nn.Linear(width, 1000))

    return nn.Sequential(*layers).to(device)


def build_loader(args, rank):
    # Generate synthetic batches with a separate stream on each rank.
    torch.manual_seed(1000 + rank)
    return DataLoader(
        SyntheticData(args.width, classes=1000),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        multiprocessing_context="spawn" if args.workers else None,
        worker_init_fn=init_worker,
    )


def start_checkpoint_writer(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    idle = context.Event()
    idle.set()
    writer = context.Process(
        target=write_checkpoints,
        args=(queue, idle, output_dir),
        name="checkpoint-writer",
        daemon=True,
    )
    writer.start()
    return writer, queue, idle


def queue_checkpoint(step, model, writer, queue, idle):
    if not writer.is_alive():
        raise RuntimeError("Checkpoint writer exited unexpectedly")
    # Skip this checkpoint if the writer is still busy.
    if idle.is_set():
        idle.clear()
        log(f"Checkpoint CPU snapshot starts: step={step}")
        weights = {
            name: tensor.detach().to("cpu", copy=True)
            for name, tensor in model.module.state_dict().items()
        }
        queue.put((step, weights))


def train(model, device, args, rank):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    loader = build_loader(args, rank)

    # Global rank zero owns the background checkpoint writer.
    if rank == 0:
        writer, queue, idle = start_checkpoint_writer(args.output_dir)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        log(
            f"Training starts: parameters={parameter_count:,} "
            f"batch_per_gpu={args.batch_size} workers_per_rank={args.workers} "
            f"checkpoint_writer_pid={writer.pid}"
        )

    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = criterion(model(inputs), targets)

        loss.backward()
        optimizer.step()

        if rank == 0 and (step == 1 or step % 10 == 0):
            log(f"step={step}/{args.steps} loss={loss.item():.6f}")

        if rank == 0 and (step == 1 or step % args.checkpoint_every == 0):
            queue_checkpoint(step, model, writer, queue, idle)

        if step >= args.steps:
            break

    # Drain the writer before shutting down the process group.
    if rank == 0:
        queue.put(None)
        writer.join()
        log(f"Training completed; checkpoint writer exit={writer.exitcode}")


def main():
    args = parse_args()

    # torchrun assigns a global rank and a local GPU index.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    log(f"Training rank={rank} gpu={torch.cuda.get_device_name(device)}")

    model = DDP(build_model(args.width, args.depth, device), device_ids=[local_rank])
    train(model, device, args, rank)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
