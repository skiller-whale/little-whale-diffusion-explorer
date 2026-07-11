import * as ort from "onnxruntime-web/webgpu";
import manifest from "../public/models/manifest.json";
import { ddimStep, gaussianNoise, inferenceTimesteps, timestepEmbedding, type PredictionType } from "../src/diffusion";

ort.env.wasm.numThreads = 1;
for (const config of manifest.models) {
  const bytes = new Uint8Array(await Bun.file(`public/models/${config.model}`).arrayBuffer());
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  let sample = gaussianNoise(3 * config.imageSize ** 2, 24601);
  const times = inferenceTimesteps(20), started = performance.now();
  for (let index = 0; index < times.length; index++) {
    const timestep = times[index];
    const output = await session.run({
      sample: new ort.Tensor("float32", sample, [1, 3, config.imageSize, config.imageSize]),
      timestep_embedding: new ort.Tensor("float32", timestepEmbedding(timestep, config.timeEmbeddingSize), [1, config.timeEmbeddingSize]),
    });
    const prediction = output[config.outputName]?.data;
    if (!(prediction instanceof Float32Array) || prediction.length !== 3 * config.imageSize ** 2) throw new Error(`${config.imageSize}px model violates its browser contract`);
    sample = ddimStep(sample, prediction, timestep, times[index + 1] ?? -1, config.predictionType as PredictionType);
  }
  if (!sample.every(Number.isFinite)) throw new Error(`${config.imageSize}px generation returned non-finite pixels`);
  console.log(`${config.imageSize}×${config.imageSize} WASM generation passed in ${(performance.now() - started).toFixed(0)} ms`);
}
