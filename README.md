# Little Whale Diffusion

A compact React demo that turns seeded noise into 64×64 orca illustrations with a diffusion model running entirely in the browser. WebGPU is preferred when available and ONNX Runtime's WASM backend is the automatic fallback.

## Run the app

```bash
bun install
bun dev
```

Open the printed local URL, wait for the 16.3 MB model to load, then choose a seed and click **Denoise a whale**. The timeline caches every denoising frame, so completed steps can be inspected without rerunning the model.

Production verification:

```bash
bun test
bun run typecheck
bun run build
bun scripts/smoke-model.ts
```

The production output in `dist/` is static and needs no backend. ONNX Runtime's WASM files are copied into the build so inference has no CDN dependency.

## Train and export

The dataset is generated from code: it renders orca-only anatomy with larger curved silhouettes, distinct dorsal and pectoral fins, tail flukes, separate eye and saddle patches, restrained white belly markings, curated poses and ocean palettes, and sparse environmental detail. It contains no downloaded or third-party imagery.

```bash
python3 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements.txt
training/.venv/bin/python training/train.py preview
training/.venv/bin/python training/train.py train
training/.venv/bin/python training/train.py sample training/checkpoints/whale-200000.pt
training/.venv/bin/python training/train.py export training/checkpoints/whale-200000.pt
```

The generator supports a deterministic 150,000-image corpus. Training defaults to a 90/10 split, 200,000 optimization steps, a 500-step warmup with cosine learning-rate decay, mixed precision on supported accelerators, and EMA weights. `--run-name` keeps experiments isolated, and checkpoints can be resumed with `--resume PATH`. Run `training/train.py --help` and the subcommand help for smaller curriculum runs and device selection.

The model is a seed-conditioned velocity-predicting DDPM U-Net with a cosine noise schedule. The seed selects one of 16 learned orca archetypes while its Gaussian noise still controls individual variation. Velocity prediction remains stable near pure noise without dividing by a tiny signal coefficient. Browser inference uses deterministic DDIM (`eta = 0`) at 10, 20, 30, or 50 steps. Its ONNX interface is intentionally static:

| Name | Direction | Type and shape |
| --- | --- | --- |
| `sample` | input | float32 `[1, 3, 64, 64]` |
| `timestep_embedding` | input | float32 `[1, 192]` |
| `conditioning` | input | float32 `[1, 16]` one-hot archetype |
| `predicted_velocity` | output | float32 `[1, 3, 64, 64]` |

The checked-in model uses EMA weights from a 2,048-image conditioned curriculum. Checkpoints were compared at the browser's default 20 DDIM steps; the 5,000-step checkpoint was selected over 2,500 and 7,500 because it gave the best balance of anatomy, markings, brightness, and seed coverage. Its exact checksum and scheduler contract are recorded in `public/models/manifest.json`.
