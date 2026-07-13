"""Capture FLUX.1-schnell denoising trajectories for the final lesson block.

Twelve variants (two step-count ladders plus four wilder prompts), each saved as
the full per-step frame sequence so the UI can scrub any of them like the little
whale generators.

Run manually (never in CI) with the mflux venv:
    training/.venv-mflux/bin/python training/generate_flux_frames.py
mflux is installed separately from requirements.txt: python3 -m venv training/.venv-mflux
&& training/.venv-mflux/bin/pip install mflux. First run downloads ~17GB of weights.
Variants are cached under training/outputs/flux_steps/<id>; delete a directory to
regenerate just that variant.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# The official black-forest-labs repo is gated on HuggingFace; this is an ungated
# mirror of the same Apache-2.0 weights, pre-quantized to 8-bit in mflux format.
MODEL = "dhairyashil/FLUX.1-schnell-mflux-8bit"
BASE_MODEL = "schnell"
MODEL_LABEL = "FLUX.1 schnell"
RENDER_SIZE = 768
OUTPUT_SIZE = 512
WEBP_QUALITY = 82

WATERCOLOR_PROMPT = "a watercolor illustration of an orca whale swimming through a deep teal ocean"
PHOTO_PROMPT = "a photograph of an orca breaching beside a small fishing boat at dawn, spray catching the light"
STEP_LADDER = (1, 2, 4, 20)
WILD_STEPS = 4  # schnell is step-distilled; 4 steps is the model's recommended count
WILD_PROMPTS = (
    ("stained-glass", "glass", "an orca made of stained glass, backlit in a cathedral window", 11),
    ("blueprint", "blueprint", "a blueprint schematic of a mechanical orca, white lines on blue paper", 23),
    ("origami", "origami", "an origami orca folded from old nautical charts, studio photograph", 31),
    ("neon-city", "neon city", "a neon-lit orca gliding above a rain-slick city street at night", 55),
)

STEPWISE_ROOT = Path(__file__).parent / "outputs" / "flux_steps"
OUTPUT = Path(__file__).parents[1] / "public" / "flux"


def ladder(prefix: str, prompt: str, seed: int) -> list[dict]:
    return [{"id": f"{prefix}-{steps}", "label": f"{steps} step{'s' if steps > 1 else ''}",
             "prompt": prompt, "seed": seed, "steps": steps} for steps in STEP_LADDER]


def build_rows() -> list[dict]:
    return [
        {"title": "Watercolour orca at various step counts", "variants": ladder("watercolor", WATERCOLOR_PROMPT, 42)},
        {"title": "Photograph orca at various step counts", "variants": ladder("photo", PHOTO_PROMPT, 7)},
        {"title": "Various prompts at 4 steps", "variants": [
            {"id": id, "label": label, "prompt": prompt, "seed": seed, "steps": WILD_STEPS}
            for id, label, prompt, seed in WILD_PROMPTS
        ]},
    ]


def step_frames(directory: Path) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = re.search(r"step(\d+)of", path.name)
            if match:
                frames.append((int(match.group(1)), path))
    return sorted(frames)


def generate_variant(variant: dict, index: int, count: int) -> None:
    directory = STEPWISE_ROOT / variant["id"]
    if len(step_frames(directory)) >= variant["steps"] + 1:
        print(f"[{index + 1}/{count}] {variant['id']}: cached")
        return
    print(f"[{index + 1}/{count}] {variant['id']}: {variant['steps']} steps · {variant['prompt']}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    subprocess.run([
        str(Path(sys.executable).parent / "mflux-generate"),
        "--model", MODEL, "--base-model", BASE_MODEL,
        "--prompt", variant["prompt"], "--seed", str(variant["seed"]), "--steps", str(variant["steps"]),
        "--width", str(RENDER_SIZE), "--height", str(RENDER_SIZE),
        "--stepwise-image-output-dir", str(directory),
        "--output", str(directory / "final.png"),
    ], check=True)


def noise_frame(seed: int) -> Image.Image:
    sample = np.random.default_rng(seed).standard_normal((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.float32)
    return Image.fromarray(np.clip((sample + 1) * 127.5, 0, 255).astype(np.uint8), "RGB")


def export_variant(variant: dict) -> None:
    frames = step_frames(STEPWISE_ROOT / variant["id"])
    images = [Image.open(path).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS) for _, path in frames]
    if frames[0][0] != 0:
        images.insert(0, noise_frame(variant["seed"]))
    variant["frameCount"] = len(images)
    directory = OUTPUT / variant["id"]
    directory.mkdir(parents=True)
    for index, image in enumerate(images):
        image.save(directory / f"{index:03d}.webp", "WEBP", quality=WEBP_QUALITY)


def main() -> None:
    rows = build_rows()
    variants = [variant for row in rows for variant in row["variants"]]
    for index, variant in enumerate(variants):
        generate_variant(variant, index, len(variants))
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for variant in variants:
        export_variant(variant)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "model": MODEL_LABEL, "size": OUTPUT_SIZE, "rows": rows,
    }, indent=2) + "\n")
    total = sum(path.stat().st_size for path in OUTPUT.rglob("*.webp"))
    frame_count = sum(variant["frameCount"] for variant in variants)
    print(f"wrote {frame_count} frames across {len(variants)} variants to {OUTPUT} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
