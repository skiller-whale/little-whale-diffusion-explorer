"""Export static examples from the exact generator used to train the 32 px model."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from PIL import Image
from dataset_32 import render_whale

TIMESTEPS = (0, 70, 180, 330, 500, 680, 840, 999)
OUTPUT = Path(__file__).parents[1] / "public" / "training" / "32"

def alpha_cumprod(timestep: int, steps: int = 1000, offset: float = .008) -> float:
    f0 = math.cos(offset / (1 + offset) * math.pi * .5) ** 2
    x = (timestep + 1) / steps
    return max(math.cos((x + offset) / (1 + offset) * math.pi * .5) ** 2 / f0, 1e-7)

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for row in range(5):
        clean = np.asarray(render_whale(row), dtype=np.float32) / 127.5 - 1
        noise = np.random.default_rng(81_000 + row).standard_normal(clean.shape, dtype=np.float32)
        for column, timestep in enumerate(TIMESTEPS):
            if column == 0:
                sample = clean
            else:
                alpha = alpha_cumprod(timestep)
                sample = math.sqrt(alpha) * clean + math.sqrt(1 - alpha) * noise
            pixels = np.clip((sample + 1) * 127.5, 0, 255).astype(np.uint8)
            Image.fromarray(pixels, "RGB").save(OUTPUT / f"{row}-{column}.png", optimize=True)

if __name__ == "__main__":
    main()
