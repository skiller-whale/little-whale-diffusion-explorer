#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import WhaleDataset, WhaleDatasetConfig, render_whale
from model import WhaleUNet

ROOT = Path(__file__).resolve().parents[1]


def embedding(timesteps: torch.Tensor, size: int = 192) -> torch.Tensor:
    half = size // 2
    frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=timesteps.device) / (half - 1))
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)


def cosine_alphas(steps: int = 1_000, offset: float = .008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1)
    values = torch.cos(((x / steps + offset) / (1 + offset)) * math.pi * .5) ** 2
    values = values / values[0]
    betas = 1 - values[1:] / values[:-1]
    return torch.cumprod(1 - betas.clamp(0, .999), dim=0)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except RuntimeError:
            pass
    return torch.device("cpu")


def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.lerp_(source, 1 - decay)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dataset = WhaleDataset(WhaleDatasetConfig(length=args.dataset_size, seed=args.seed))
    train_size = int(len(dataset) * .9)
    training, _ = random_split(dataset, (train_size, len(dataset) - train_size), generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    model = WhaleUNet().to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return max(step + 1, 1) / max(args.warmup_steps, 1)
        progress = min(1.0, (step - args.warmup_steps) / max(args.train_steps - args.warmup_steps, 1))
        minimum = args.min_learning_rate / args.learning_rate
        return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    alphas = cosine_alphas().to(device)
    checkpoint_dir = ROOT / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"]); optimizer.load_state_dict(state["optimizer"]); start = state["step"]
        if "scheduler" in state: scheduler.load_state_dict(state["scheduler"])

    iterator = iter(loader)
    recent_losses: deque[float] = deque(maxlen=100)
    model.train()
    for step in range(start + 1, args.train_steps + 1):
        try: clean = next(iterator)
        except StopIteration: iterator = iter(loader); clean = next(iterator)
        clean = clean.to(device)
        times = torch.randint(0, 1_000, (clean.shape[0],), device=device)
        noise = torch.randn_like(clean)
        alpha = alphas[times, None, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type in ("cuda", "mps")):
            velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
            loss = torch.mean((model(noisy, embedding(times)) - velocity) ** 2)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        # Warm EMA up instead of letting its random initialization dominate a short run.
        update_ema(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))
        recent_losses.append(loss.item())
        if step == 1 or step % args.log_every == 0:
            average_loss = sum(recent_losses) / len(recent_losses)
            print(f"step={step} loss={average_loss:.5f} lr={optimizer.param_groups[0]['lr']:.2e} device={device}", flush=True)
        if step % args.save_every == 0 or step == args.train_steps:
            path = checkpoint_dir / f"{args.run_name}-{step:06d}.pt"
            saved_args = {key: value for key, value in vars(args).items() if key != "function"}
            torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "args": saved_args}, path)
            print(f"saved {path}")


def export(args: argparse.Namespace) -> None:
    model = WhaleUNet().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = state[args.weights] if args.weights in state else state
    model.load_state_dict(weights)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (torch.randn(1, 3, 64, 64), torch.randn(1, 192)), output,
            input_names=["sample", "timestep_embedding"], output_names=["predicted_velocity"],
            opset_version=18, dynamo=False, do_constant_folding=True,
        )
    size = output.stat().st_size
    if size > 30 * 1024 * 1024: raise RuntimeError(f"Model is {size / 1024**2:.1f} MB; the limit is 30 MB")
    print(f"exported {output} ({size / 1024**2:.1f} MB)")


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model = WhaleUNet().to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state[args.weights])
    generator = torch.Generator(device=device).manual_seed(args.seed)
    current = torch.randn(args.count, 3, 64, 64, device=device, generator=generator)
    alphas = cosine_alphas().to(device)
    times = torch.linspace(args.start_timestep, 0, args.steps).round().long().tolist()
    for index, timestep in enumerate(times):
        previous = times[index + 1] if index + 1 < len(times) else -1
        time = torch.full((args.count,), timestep, device=device)
        velocity = model(current, embedding(time))
        alpha = alphas[timestep]
        previous_alpha = alphas[previous] if previous >= 0 else torch.tensor(1., device=device)
        clean = (alpha.sqrt() * current - (1-alpha).sqrt() * velocity).clamp(-1, 1)
        noise = (1-alpha).sqrt() * current + alpha.sqrt() * velocity
        current = previous_alpha.sqrt() * clean + (1-previous_alpha).sqrt() * noise
    pixels = ((current.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    columns = math.ceil(math.sqrt(args.count)); rows = math.ceil(args.count / columns)
    sheet = Image.new("RGB", (columns * 64, rows * 64))
    for index, array in enumerate(pixels): sheet.paste(Image.fromarray(array), ((index % columns) * 64, (index // columns) * 64))
    sheet = sheet.resize((columns * 192, rows * 192), Image.Resampling.NEAREST)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); sheet.save(output)
    print(f"wrote {output}")


def preview(args: argparse.Namespace) -> None:
    rows, columns, size = 4, 4, 64
    sheet = Image.new("RGB", (columns * size, rows * size))
    for index in range(rows * columns): sheet.paste(render_whale(args.seed + index), ((index % columns) * size, (index // columns) * size))
    sheet = sheet.resize((columns * size * 4, rows * size * 4), Image.Resampling.NEAREST)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); sheet.save(output)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export Little Whale Diffusion")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--device", default="auto"); train_parser.add_argument("--seed", type=int, default=24_601)
    train_parser.add_argument("--dataset-size", type=int, default=150_000); train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--workers", type=int, default=4); train_parser.add_argument("--train-steps", type=int, default=200_000)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4); train_parser.add_argument("--ema-decay", type=float, default=.9999)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-5); train_parser.add_argument("--warmup-steps", type=int, default=500); train_parser.add_argument("--run-name", default="whale")
    train_parser.add_argument("--log-every", type=int, default=100); train_parser.add_argument("--save-every", type=int, default=10_000); train_parser.add_argument("--resume")
    train_parser.set_defaults(function=train)
    export_parser = commands.add_parser("export"); export_parser.add_argument("checkpoint"); export_parser.add_argument("--weights", choices=("model", "ema"), default="ema"); export_parser.add_argument("--output", default=ROOT / "public/models/whale-ddpm.onnx"); export_parser.set_defaults(function=export)
    preview_parser = commands.add_parser("preview"); preview_parser.add_argument("--seed", type=int, default=0); preview_parser.add_argument("--output", default=ROOT / "training/runs/dataset-preview.png"); preview_parser.set_defaults(function=preview)
    sample_parser = commands.add_parser("sample"); sample_parser.add_argument("checkpoint"); sample_parser.add_argument("--weights", choices=("model", "ema"), default="ema"); sample_parser.add_argument("--device", default="auto"); sample_parser.add_argument("--seed", type=int, default=24601); sample_parser.add_argument("--count", type=int, default=16); sample_parser.add_argument("--steps", type=int, default=50); sample_parser.add_argument("--start-timestep", type=int, default=999); sample_parser.add_argument("--output", default=ROOT / "training/runs/model-samples.png"); sample_parser.set_defaults(function=sample)
    args = parser.parse_args(); args.function(args)


if __name__ == "__main__": main()
