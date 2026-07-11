"""Deterministic, license-free 64 px orca illustration synthesis."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


OCEANS = [
    ((3, 24, 39), (7, 65, 82), (41, 132, 143)),
    ((5, 25, 48), (15, 68, 105), (66, 143, 166)),
    ((4, 34, 43), (10, 82, 87), (62, 151, 139)),
    ((12, 23, 54), (31, 65, 112), (89, 137, 169)),
]
ORCA_ARCHETYPES = 16


@dataclass(frozen=True)
class WhaleDatasetConfig:
    length: int = 150_000
    size: int = 64
    render_scale: int = 4
    seed: int = 24_601


def _gradient(image: Image.Image, top: tuple[int, ...], bottom: tuple[int, ...]) -> None:
    draw = ImageDraw.Draw(image)
    for y in range(image.height):
        mix = y / max(image.height - 1, 1)
        color = tuple(round(a * (1 - mix) + b * mix) for a, b in zip(top, bottom))
        draw.line((0, y, image.width, y), fill=color)


def _orca_layer(rng: random.Random, archetype: int, extent: int, accent: tuple[int, ...]) -> Image.Image:
    """Build a side-on orca from a curved silhouette and characteristic markings."""
    layer = Image.new("RGBA", (extent, extent))
    draw = ImageDraw.Draw(layer, "RGBA")
    cx, cy = extent * .5, extent * .5
    identity = random.Random(71_071 + archetype * 9_973)
    length = identity.uniform(.63, .73) * extent * rng.uniform(.97, 1.03)
    height = identity.uniform(.205, .245) * extent * rng.uniform(.97, 1.03)
    nose = cx + length * .47
    tail_root = cx - length * .40

    # Organic body outline: blunt melon, tapered peduncle and curved back/belly.
    points: list[tuple[float, float]] = []
    for i in range(31):
        u = i / 30
        x = tail_root + (nose - tail_root) * u
        profile = math.sin(math.pi * u) ** .58
        head = 1 + .18 * math.exp(-((u - .84) / .16) ** 2)
        y = cy - height * .5 * profile * head * (1 + rng.uniform(-.025, .025))
        points.append((x, y))
    for i in range(30, -1, -1):
        u = i / 30
        x = tail_root + (nose - tail_root) * u
        profile = math.sin(math.pi * u) ** .68
        belly = 1 + .16 * math.exp(-((u - .58) / .22) ** 2)
        y = cy + height * .5 * profile * belly * (1 + rng.uniform(-.025, .025))
        points.append((x, y))
    black = rng.choice(((5, 14, 19, 255), (8, 18, 25, 255), (10, 20, 28, 255)))
    draw.polygon(points, fill=black)

    # Symmetrical tail flukes and a species-defining tall dorsal fin.
    tail = length * rng.uniform(.12, .16)
    draw.polygon([(tail_root + length*.025, cy), (tail_root-tail*.95, cy-tail*.68),
                  (tail_root-tail*.78, cy-tail*.05), (tail_root-tail*.92, cy+tail*.62)], fill=black)
    dorsal_x = cx - length * identity.uniform(.05, .13)
    dorsal_h = height * identity.uniform(.82, 1.12)
    draw.polygon([(dorsal_x-height*.13, cy-height*.42), (dorsal_x+height*.05, cy-dorsal_h),
                  (dorsal_x+height*.23, cy-height*.38)], fill=black)

    # Pectoral fin, white belly field, eye patch and grey saddle patch.
    fin_x = cx + length * rng.uniform(.01, .13)
    draw.polygon([(fin_x-height*.13, cy+height*.28), (fin_x+height*.03, cy+height*1.02),
                  (fin_x+height*.42, cy+height*.34)], fill=(3, 11, 16, 245))
    white = identity.choice(((229, 242, 236, 255), (210, 232, 229, 255), (239, 242, 226, 255)))
    draw.ellipse((cx-length*.02, cy+height*.16, nose-length*.09, cy+height*.43), fill=white)
    patch_x = nose - length * identity.uniform(.18, .22)
    patch_y = cy - height * identity.uniform(.24, .31)
    draw.ellipse((patch_x, patch_y, patch_x+length*.075, patch_y+height*.15), fill=white)
    saddle = tuple(round(c*.58 + a*.42) for c, a in zip(black[:3], accent)) + (185,)
    draw.ellipse((cx-length*.22, cy-height*.48, cx+length*.02, cy-height*.22), fill=saddle)

    # Subtle highlight along the back reads well after downsampling.
    draw.line(points[:25], fill=(*accent, 55), width=max(1, extent // 128))
    return layer


def render_whale(index: int, config: WhaleDatasetConfig = WhaleDatasetConfig()) -> Image.Image:
    rng = random.Random(config.seed + index * 1_000_003)
    archetype = index % ORCA_ARCHETYPES
    size = config.size * config.render_scale
    top, bottom, accent = rng.choice(OCEANS)
    image = Image.new("RGB", (size, size))
    _gradient(image, top, bottom)
    environment = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(environment, "RGBA")

    # Environmental detail stays subordinate to the single-orca composition.
    for _ in range(rng.randint(0, 2)):
        x = rng.randint(-size//4, size)
        spread = rng.randint(size//14, size//5)
        draw.polygon([(x, 0), (x+spread, 0), (x+spread*2, size), (x-spread, size)], fill=(120, 220, 214, rng.randint(3, 8)))
    for _ in range(rng.randint(0, 2)):
        x = rng.randrange(size)
        kelp_top = rng.uniform(.62, .86) * size
        draw.line([(x, size), (x+rng.uniform(-.04,.04)*size, kelp_top)], fill=(3, 39, 39, rng.randint(50, 100)), width=max(2, size//80))
    for _ in range(rng.randint(0, 5)):
        x, y = rng.randrange(size), rng.randrange(size)
        radius = rng.uniform(.002, .012) * size
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=(*accent, rng.randint(20, 55)), width=max(1, config.render_scale//2))
    environment = environment.filter(ImageFilter.GaussianBlur(rng.uniform(.2, .65) * config.render_scale))
    image = Image.alpha_composite(image.convert("RGBA"), environment)

    orca = _orca_layer(rng, archetype, size, accent)
    identity = random.Random(91_091 + archetype * 8_191)
    angle = identity.uniform(-8, 8) + rng.uniform(-2, 2)
    orca = orca.rotate(angle, resample=Image.Resampling.BICUBIC, center=(size/2, size/2))
    scale = rng.uniform(.94, 1.03)
    if scale != 1:
        resized = orca.resize((round(size*scale), round(size*scale)), Image.Resampling.LANCZOS)
        positioned = Image.new("RGBA", (size, size))
        positioned.alpha_composite(resized, ((size-resized.width)//2, (size-resized.height)//2))
        orca = positioned
    if rng.random() < .5: orca = orca.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    offset = (rng.randint(-size//30, size//30), rng.randint(-size//28, size//28))
    image.alpha_composite(orca, offset)
    image = image.convert("RGB").resize((config.size, config.size), Image.Resampling.LANCZOS)
    return image


class WhaleDataset(Dataset):
    def __init__(self, config: WhaleDatasetConfig = WhaleDatasetConfig()): self.config = config
    def __len__(self) -> int: return self.config.length
    def __getitem__(self, index: int):
        import torch
        pixels = np.asarray(render_whale(index, self.config), dtype=np.float32) / 127.5 - 1.0
        condition = torch.zeros(ORCA_ARCHETYPES, dtype=torch.float32)
        condition[index % ORCA_ARCHETYPES] = 1
        return torch.from_numpy(pixels).permute(2, 0, 1), condition
