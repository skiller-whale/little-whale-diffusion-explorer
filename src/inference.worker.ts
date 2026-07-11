/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import { CHANNELS, ddimStep, gaussianNoise, inferenceTimesteps, tensorToRgba, timestepEmbedding, type ModelConfig } from "./diffusion";
import type { WorkerRequest, WorkerResponse } from "./protocol";

const context: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;
let session: ort.InferenceSession | undefined;
let config: ModelConfig | undefined;
let generation = 0;
const send = (message: WorkerResponse, transfer: Transferable[] = []) => context.postMessage(message, transfer);

async function initialize(next: ModelConfig) {
  config = next;
  ort.env.wasm.wasmPaths = `${location.origin}/ort/`;
  ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1)) : 1;
  send({ type: "status", message: "Loading model" });
  if ("gpu" in navigator) {
    try { session = await ort.InferenceSession.create(next.modelUrl, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" }); }
    catch (error) { console.warn("WebGPU initialization failed; using WASM", error); }
  }
  session ??= await ort.InferenceSession.create(next.modelUrl, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  send({ type: "ready" });
}

async function generate(id: number, seed: number, steps: number) {
  if (!session || !config) throw new Error("The model is not ready");
  const token = ++generation;
  const started = performance.now(), size = config.imageSize;
  const shape = [1, CHANNELS, size, size];
  let sample = gaussianNoise(CHANNELS * size * size, seed);
  const times = inferenceTimesteps(steps);
  let pixels = tensorToRgba(sample, size);
  send({ type: "frame", id, index: 0, total: steps, pixels, elapsedMs: 0 }, [pixels.buffer]);
  for (let index = 0; index < times.length; index++) {
    if (token !== generation) { send({ type: "cancelled", id }); return; }
    const timestep = times[index];
    const result = await session.run({
      sample: new ort.Tensor("float32", sample, shape),
      timestep_embedding: new ort.Tensor("float32", timestepEmbedding(timestep, config.timeEmbeddingSize), [1, config.timeEmbeddingSize]),
    });
    const prediction = result[config.outputName]?.data;
    if (!(prediction instanceof Float32Array)) throw new Error("The model returned an invalid tensor");
    sample = ddimStep(sample, prediction, timestep, times[index + 1] ?? -1, config.predictionType);
    pixels = tensorToRgba(sample, size);
    send({ type: "frame", id, index: index + 1, total: steps, pixels, elapsedMs: performance.now() - started }, [pixels.buffer]);
  }
  if (token === generation) send({ type: "complete", id, elapsedMs: performance.now() - started });
}

context.onmessage = ({ data }: MessageEvent<WorkerRequest>) => {
  if (data.type === "initialize") void initialize(data.config).catch((error) => send({ type: "error", message: String(error?.message ?? error) }));
  if (data.type === "generate") void generate(data.id, data.seed, data.steps).catch((error) => send({ type: "error", message: String(error?.message ?? error) }));
  if (data.type === "cancel") generation++;
};
