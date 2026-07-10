# Little Whale Diffusion

A compact React demo that turns seeded noise into 32×32 whale icons with a diffusion model running entirely in the browser. WebGPU is preferred when available and ONNX Runtime's WASM backend is the automatic fallback.

## Run the app

```bash
bun install
bun dev
```

Open the printed local URL, wait for the 6.8 MB model to load, then choose a seed and click **Denoise a whale**. The timeline caches every denoising frame, so completed steps can be inspected without rerunning the model.

Production verification:

```bash
bun test
bun run typecheck
bun run build
bun scripts/smoke-model.ts
```

The production output in `dist/` is static and needs no backend. ONNX Runtime's WASM files are copied into the build so inference has no CDN dependency.

## Train and export

The dataset is generated from code: it renders varied whale silhouettes, fins, tails, eyes, spouts, bubbles, palettes, and ocean backgrounds. It contains no downloaded or third-party imagery.

```bash
python3 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements.txt
training/.venv/bin/python training/train.py preview
training/.venv/bin/python training/train.py train
training/.venv/bin/python training/train.py sample training/checkpoints/whale-200000.pt
training/.venv/bin/python training/train.py export training/checkpoints/whale-200000.pt
```

Training defaults to a 50,000-image deterministic dataset, a 90/10 split, 200,000 optimization steps, mixed precision on supported accelerators, and EMA weights. Checkpoints can be resumed with `--resume PATH`. Run `training/train.py --help` and the subcommand help for smaller verification runs and device selection.

The model is an unconditional `x₀`-predicting DDPM U-Net with a cosine noise schedule. Browser inference uses deterministic DDIM (`eta = 0`) at 10, 20, 30, or 50 steps. Its ONNX interface is intentionally static:

| Name | Direction | Type and shape |
| --- | --- | --- |
| `sample` | input | float32 `[1, 3, 32, 32]` |
| `timestep_embedding` | input | float32 `[1, 160]` |
| `predicted_clean` | output | float32 `[1, 3, 32, 32]` |

The checked-in model was trained as a compact local verification run and uses warmed EMA weights. Its exact checksum and scheduler contract are recorded in `public/models/manifest.json`; rerun the longer default recipe when developing higher-quality weights.
