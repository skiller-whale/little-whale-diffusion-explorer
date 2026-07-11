# Little Whale Diffusion

A lightweight Bun and React teaching demo for two unconditional diffusion models running entirely in the browser. It compares the original 32×32 clean-image-predicting model with the newer 64×64 velocity-predicting model, and illustrates the forward-noising process used during training.

## Run locally

```bash
bun install
bun dev
```

Both models load when the page opens. Each training illustration precedes its generator. Seed changes and committed step-slider changes regenerate and cache the complete denoising timeline in the background; **Play** plays the cached frames, and **Save** collects native-resolution copies of completed results beneath the generator.

## Verify

```bash
bun test
bun run typecheck
bun run build
bun scripts/smoke-model.ts
training/.venv/bin/python -m unittest discover -s training
```

The checked-in manifests pin each ONNX checksum and inference contract. WebGPU is attempted when available, with ONNX Runtime WASM as the automatic fallback.

## Models

Both models use a 1,000-step cosine noise schedule and deterministic DDIM browser sampling. They are strictly unconditional: inference receives only seeded Gaussian noise and a timestep embedding.

- 32×32: original approximately 6.8 MB model, 160-value time embedding, direct clean-image prediction.
- 64×64: approximately 16.3 MB model, 192-value time embedding, velocity prediction.

The 32×32 table is exported from the exact historical procedural generator that trained the original model. The training tables use the same cosine forward-noising equation as their models. Training and export tooling lives in `training/`; regenerate the historical table with `training/.venv/bin/python training/export_training_examples.py`.
