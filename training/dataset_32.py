"""Deterministic, license-free whale icon synthesis."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


PALETTES = [
    ((7, 35, 53), (50, 124, 151), (148, 214, 215)),
    ((15, 42, 61), (47, 93, 142), (178, 212, 228)),
    ((8, 48, 58), (52, 143, 137), (176, 226, 205)),
    ((24, 34, 64), (83, 91, 164), (196, 196, 235)),
    ((10, 42, 48), (46, 111, 124), (213, 231, 210)),
]


@dataclass(frozen=True)
class WhaleDatasetConfig:
    length: int = 50_000
    size: int = 32
    render_scale: int = 4
    seed: int = 24_601


def _ellipse_point(cx: float, cy: float, rx: float, ry: float, angle: float) -> tuple[float, float]:
    return cx + math.cos(angle) * rx, cy + math.sin(angle) * ry


def render_whale(index: int, config: WhaleDatasetConfig = WhaleDatasetConfig()) -> Image.Image:
    rng = random.Random(config.seed + index * 1_000_003)
    scale = config.render_scale
    width = height = config.size * scale
    background, body, highlight = rng.choice(PALETTES)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image, "RGBA")

    # Quiet ocean depth and particulate bubbles.
    for y in range(height):
        shade = int(13 * y / height)
        draw.line((0, y, width, y), fill=(*tuple(min(255, c + shade) for c in background), 255))
    for _ in range(rng.randint(4, 11)):
        x, y = rng.randrange(width), rng.randrange(height)
        r = rng.choice((1, 2, 3, 4)) * scale / 2
        draw.ellipse((x-r, y-r, x+r, y+r), outline=(*highlight, rng.randint(45, 110)), width=max(1, scale // 2))

    facing = rng.choice((-1, 1))
    cx = rng.uniform(.48, .55) * width
    cy = rng.uniform(.49, .57) * height
    rx = rng.uniform(.25, .31) * width
    ry = rng.uniform(.13, .18) * height
    body_box = (cx-rx, cy-ry, cx+rx, cy+ry)
    draw.ellipse(body_box, fill=(*body, 255))

    # Tail points away from the rounded head.
    tail_x = cx - facing * rx * .82
    tail_root_y = cy
    fin = rng.uniform(.12, .17) * width
    tail_color = tuple(max(0, c - 7) for c in body)
    draw.polygon([
        (tail_x, tail_root_y), (tail_x-facing*fin*.78, tail_root_y-fin),
        (tail_x-facing*fin*.9, tail_root_y-fin*.18),
        (tail_x-facing*fin*.86, tail_root_y+fin*.82),
    ], fill=(*tail_color, 255))

    # Belly, flipper and eye establish a readable silhouette at 32 px.
    belly = (cx-rx*.25, cy+ry*.1, cx+rx*facing*.85, cy+ry*.78)
    draw.ellipse((min(belly[0], belly[2]), belly[1], max(belly[0], belly[2]), belly[3]), fill=(*highlight, 120))
    flipper_x = cx + facing * rx * .05
    draw.polygon([(flipper_x, cy+ry*.45), (flipper_x-facing*rx*.08, cy+ry*1.2), (flipper_x+facing*rx*.38, cy+ry*.55)], fill=(*tail_color, 235))
    eye_x = cx + facing * rx * .57
    eye_y = cy - ry * .25
    er = max(2, scale * .75)
    draw.ellipse((eye_x-er, eye_y-er, eye_x+er, eye_y+er), fill=(4, 18, 23, 255))
    draw.ellipse((eye_x-er*.45, eye_y-er*.55, eye_x, eye_y-er*.1), fill=(240, 251, 245, 230))

    # Optional blowhole spray and a soft vignette.
    if rng.random() < .7:
        spout_x = cx + facing * rx * .2
        top = cy - ry * 1.8
        draw.arc((spout_x-fin*.35, top, spout_x, cy-ry*.6), 195, 330, fill=(*highlight, 180), width=scale)
        draw.arc((spout_x, top, spout_x+fin*.35, cy-ry*.6), 210, 345, fill=(*highlight, 150), width=scale)

    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0, .28) * scale))
    image = image.resize((config.size, config.size), Image.Resampling.LANCZOS)
    if facing < 0:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


class WhaleDataset(Dataset):
    def __init__(self, config: WhaleDatasetConfig = WhaleDatasetConfig()):
        self.config = config

    def __len__(self) -> int:
        return self.config.length

    def __getitem__(self, index: int):
        import torch
        pixels = np.asarray(render_whale(index, self.config), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(pixels).permute(2, 0, 1)
