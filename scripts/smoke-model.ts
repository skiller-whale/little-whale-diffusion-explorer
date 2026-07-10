import * as ort from "onnxruntime-web/webgpu";
import { ddimStep, gaussianNoise, inferenceTimesteps, timestepEmbedding } from "../src/diffusion";

ort.env.wasm.numThreads = 1;
const bytes = new Uint8Array(await Bun.file("public/models/whale-ddpm.onnx").arrayBuffer());
const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
let sample = gaussianNoise(3 * 32 * 32, 24601);
const times = inferenceTimesteps(20);
const started = performance.now();
for (let index = 0; index < times.length; index++) {
  const timestep = times[index];
  const output = await session.run({
    sample: new ort.Tensor("float32", sample, [1, 3, 32, 32]),
    timestep_embedding: new ort.Tensor("float32", timestepEmbedding(timestep), [1, 160]),
  });
  if (!(output.predicted_clean?.data instanceof Float32Array) || output.predicted_clean.data.length !== 3 * 32 * 32) {
    throw new Error("Model output does not satisfy the browser contract");
  }
  sample = ddimStep(sample, output.predicted_clean.data, timestep, times[index + 1] ?? -1);
}
if (!sample.every(Number.isFinite)) throw new Error("Generation returned non-finite pixels");
console.log(`WASM 20-step generation passed in ${(performance.now() - started).toFixed(0)} ms`, session.inputNames, session.outputNames);
