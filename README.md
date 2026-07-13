# Little Whale Diffusion Explorer

Live site: <https://skiller-whale.github.io/little-whale-diffusion-explorer/>

A browser-based teaching demo for a live coaching session on how diffusion models work. The page walks through the whole idea at three scales: two tiny unconditional diffusion models (32×32 and 64×64) that were trained on procedurally generated whale illustrations and run entirely in the browser, and a modern production model (FLUX.1 schnell, 2024) shown through pre-rendered denoising trajectories. The toy models run live in the page, and the FLUX section scrubs through frames captured from real runs of the full model.

## What's on the page

1. **A training table for the 32×32 model.** Five whale illustrations, each shown clean and then with increasing amounts of noise added at rising timesteps. This is genuinely what the model saw during training: the table is exported from the exact historical procedural generator that produced the original training set.
2. **The 32×32 generator.** Seeded Gaussian noise goes in, a whale comes out. Learners can scrub the denoising timeline frame by frame, change the seed, change the step count, and save results for comparison.
3. **The same pair again at 64×64**, with the training table computed live in the browser using the same forward-noising equation the model was trained with.
4. **The FLUX.1 schnell showcase.** Twelve pre-rendered denoising trajectories from a large 2024 text-to-image model, scrubbed with the same timeline control as the toy generators.

## The ideas behind it

**Show the forward process before the reverse one.** Each generator is preceded by its training table. Learners first see noise being *added* to clean images at increasing timesteps, and see that the model's training task is to undo that. When they then watch generation, it reads as the same process run backwards from pure noise, rather than as an unexplained trick.

**Keep one interaction across every scale.** The toy generators and the FLUX showcase share the same scrubbable "denoising timeline". The mental model a learner builds on a 6.8 MB whale model transfers directly to a state-of-the-art production model, because the loop really is the same one. The visible differences between the sections are scale, architecture, and conditioning, and those are exactly the things we want learners comparing.

**Make it deterministic so learners can experiment.** Noise is generated from a seed, and sampling uses deterministic DDIM. The same seed always produces the same whale. This matters pedagogically for two reasons. It lets learners vary one thing at a time — fix the seed and change the step count, or fix the steps and change the seed — and see exactly what each variable does. And it helps dismantle the idea that the model is retrieving images from somewhere: there is no image of this whale anywhere; the seed and the weights fully determine it.

**Keep the models small enough to feel knowable.** The 32×32 model is about 6.8 MB and runs in a web worker on the learner's own machine. The complete pipeline — dataset generator, training loop, exported ONNX model, browser sampler — is in this repository and short enough to read in a sitting.

**Use comparison as the teaching instrument.** The two toy models differ in resolution, size, and prediction target (the 32×32 model predicts the clean image directly; the 64×64 model predicts velocity), so the jump between them shows what those choices change. The FLUX section then adds two comparisons of its own: step-count ladders (1, 2, 4, and 20 steps for the same prompt and seed) that make the quality-versus-compute trade-off visible, and a row of varied prompts that shows what text conditioning adds over the unconditional toys.

**Remove setup friction entirely.** The whole lesson runs from a static page. Inference uses WebGPU where available and falls back to ONNX Runtime WASM automatically, so nobody spends session time installing anything.

## Run locally

```bash
bun install
bun dev
```

Both models load when the page opens. Each training illustration precedes its generator. Seed changes and committed step-slider changes regenerate and cache the complete denoising timeline in the background; **Play** plays the cached frames, and **Save** collects native-resolution copies of completed results beneath the generator. The FLUX frames preload in the background after the page loads, so selecting a variant is instant.

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

Both toy models use a 1,000-step cosine noise schedule and deterministic DDIM browser sampling. They are strictly unconditional: inference receives only seeded Gaussian noise and a timestep embedding.

- 32×32: original approximately 6.8 MB model, 160-value time embedding, direct clean-image prediction.
- 64×64: approximately 16.3 MB model, 192-value time embedding, velocity prediction.

Training and export tooling lives in `training/`; regenerate the historical 32×32 table with `training/.venv/bin/python training/export_training_examples.py`.

## FLUX frames

The FLUX section is pre-rendered offline: running a 2024 production model in the browser is not practical, and pre-rendering keeps the live session dependent on nothing but static files. Regenerate the frames with `training/.venv-mflux/bin/python training/generate_flux_frames.py` (see the docstring in that script for setup; the first run downloads roughly 17 GB of weights). The manifest in `public/flux/manifest.json` records the prompt, seed, and step count for every variant, so each trajectory is reproducible.
