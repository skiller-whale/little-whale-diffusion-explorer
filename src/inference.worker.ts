/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import { CHANNELS, ddimStep, gaussianNoise, IMAGE_SIZE, inferenceTimesteps, seedConditioning, tensorToRgba, timestepEmbedding } from "./diffusion";
import type { Backend, WorkerRequest, WorkerResponse } from "./protocol";

const context: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;
let session: ort.InferenceSession | undefined;
let backend: Backend = "wasm";
let paused = false;
let cancelled = false;
let resume: (() => void) | undefined;

function send(message: WorkerResponse, transfer: Transferable[] = []) {
  context.postMessage(message, transfer);
}

async function createSession(modelUrl: string) {
  ort.env.wasm.wasmPaths = `${location.origin}/ort/`;
  ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1)) : 1;
  send({ type: "status", message: "Loading the tiny whale brain…" });
  if ("gpu" in navigator) {
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      backend = "webgpu";
      send({ type: "ready", backend });
      return;
    } catch (error) {
      console.warn("WebGPU initialization failed; using WASM", error);
    }
  }
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  backend = "wasm";
  send({ type: "ready", backend });
}

async function waitWhilePaused(id: number) {
  if (!paused) return;
  send({ type: "paused", id });
  await new Promise<void>((resolve) => (resume = resolve));
}

async function generate(id: number, seed: number, steps: number, frameDelayMs: number) {
  if (!session) throw new Error("The model is not ready yet");
  paused = false;
  cancelled = false;
  const started = performance.now();
  const shape = [1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE];
  let sample = gaussianNoise(CHANNELS * IMAGE_SIZE * IMAGE_SIZE, seed);
  const times = inferenceTimesteps(steps);
  let pixels = tensorToRgba(sample);
  send({ type: "frame", id, index: 0, total: steps, pixels, elapsedMs: 0 }, [pixels.buffer]);

  for (let index = 0; index < times.length; index++) {
    await waitWhilePaused(id);
    if (cancelled) {
      send({ type: "cancelled", id });
      return;
    }
    const timestep = times[index];
    const previous = index + 1 < times.length ? times[index + 1] : -1;
    const result = await session.run({
      sample: new ort.Tensor("float32", sample, shape),
      timestep_embedding: new ort.Tensor("float32", timestepEmbedding(timestep), [1, 192]),
      conditioning: new ort.Tensor("float32", seedConditioning(seed), [1, 16]),
    });
    const velocity = result.predicted_velocity?.data;
    if (!(velocity instanceof Float32Array)) throw new Error("The model returned an invalid tensor");
    sample = ddimStep(sample, velocity, timestep, previous);
    pixels = tensorToRgba(sample);
    send({ type: "frame", id, index: index + 1, total: steps, pixels, elapsedMs: performance.now() - started }, [pixels.buffer]);
    await new Promise<void>((resolve) => setTimeout(resolve, frameDelayMs));
  }
  send({ type: "complete", id, elapsedMs: performance.now() - started });
}

context.onmessage = ({ data }: MessageEvent<WorkerRequest>) => {
  if (data.type === "initialize") void createSession(data.modelUrl).catch((error) => send({ type: "error", message: String(error?.message ?? error) }));
  if (data.type === "generate") void generate(data.id, data.seed, data.steps, data.frameDelayMs).catch((error) => send({ type: "error", message: String(error?.message ?? error) }));
  if (data.type === "pause") paused = true;
  if (data.type === "resume") { paused = false; resume?.(); resume = undefined; }
  if (data.type === "cancel") { cancelled = true; paused = false; resume?.(); resume = undefined; }
};
