export const CHANNELS = 3;
export const TRAINING_STEPS = 1000;

export type PredictionType = "clean" | "velocity";

export interface ModelConfig {
  id: "orca32" | "orca64";
  label: string;
  modelUrl: string;
  assetBaseUrl: string;
  imageSize: number;
  timeEmbeddingSize: number;
  predictionType: PredictionType;
  outputName: "predicted_clean" | "predicted_velocity";
  defaultSteps: number;
}

export const MODELS: ModelConfig[] = [
  { id: "orca32", label: "32 × 32", modelUrl: `${import.meta.env.BASE_URL}models/whale-ddpm-32.onnx`, assetBaseUrl: import.meta.env.BASE_URL, imageSize: 32, timeEmbeddingSize: 160, predictionType: "clean", outputName: "predicted_clean", defaultSteps: 20 },
  { id: "orca64", label: "64 × 64", modelUrl: `${import.meta.env.BASE_URL}models/whale-ddpm-64.onnx`, assetBaseUrl: import.meta.env.BASE_URL, imageSize: 64, timeEmbeddingSize: 192, predictionType: "velocity", outputName: "predicted_velocity", defaultSteps: 20 },
];

export function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6d2b79f5) | 0;
    let t = Math.imul(value ^ (value >>> 15), 1 | value);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianNoise(length: number, seed: number): Float32Array {
  const random = mulberry32(seed);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u = Math.max(random(), Number.EPSILON);
    const v = random();
    const radius = Math.sqrt(-2 * Math.log(u));
    output[i] = radius * Math.cos(2 * Math.PI * v);
    if (i + 1 < length) output[i + 1] = radius * Math.sin(2 * Math.PI * v);
  }
  return output;
}

export function cosineAlphaCumprod(steps = TRAINING_STEPS, offset = 0.008): Float32Array {
  const values = new Float32Array(steps);
  const f0 = Math.cos((offset / (1 + offset)) * Math.PI * .5) ** 2;
  for (let t = 0; t < steps; t++) {
    const x = (t + 1) / steps;
    values[t] = Math.max(Math.cos(((x + offset) / (1 + offset)) * Math.PI * .5) ** 2 / f0, 1e-7);
  }
  return values;
}

export function inferenceTimesteps(count: number, trainingSteps = TRAINING_STEPS): number[] {
  if (count < 1) throw new Error("At least one denoising step is required");
  if (count === 1) return [trainingSteps - 1];
  return Array.from({ length: count }, (_, i) => Math.round((trainingSteps - 1) * (1 - i / (count - 1))));
}

export function timestepEmbedding(timestep: number, size: number): Float32Array {
  const half = size / 2;
  const embedding = new Float32Array(size);
  for (let i = 0; i < half; i++) {
    const frequency = Math.exp((-Math.log(10000) * i) / Math.max(half - 1, 1));
    embedding[i] = Math.cos(timestep * frequency);
    embedding[i + half] = Math.sin(timestep * frequency);
  }
  return embedding;
}

export function ddimStep(sample: Float32Array, prediction: Float32Array, timestep: number, previousTimestep: number, predictionType: PredictionType, alphas = cosineAlphaCumprod()): Float32Array {
  const alpha = alphas[timestep];
  const previousAlpha = previousTimestep >= 0 ? alphas[previousTimestep] : 1;
  const sqrtAlpha = Math.sqrt(alpha), sqrtNoise = Math.sqrt(Math.max(1 - alpha, 0));
  const sqrtPreviousAlpha = Math.sqrt(previousAlpha), sqrtPreviousNoise = Math.sqrt(Math.max(1 - previousAlpha, 0));
  const output = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++) {
    const clean = Math.max(-1, Math.min(1, predictionType === "velocity" ? sqrtAlpha * sample[i] - sqrtNoise * prediction[i] : prediction[i]));
    const noise = predictionType === "velocity" ? sqrtNoise * sample[i] + sqrtAlpha * prediction[i] : (sample[i] - sqrtAlpha * clean) / Math.max(sqrtNoise, 1e-7);
    output[i] = sqrtPreviousAlpha * clean + sqrtPreviousNoise * noise;
  }
  return output;
}

export function forwardDiffuse(clean: Float32Array, noise: Float32Array, timestep: number, alphas = cosineAlphaCumprod()): Float32Array {
  const a = Math.sqrt(alphas[timestep]), n = Math.sqrt(1 - alphas[timestep]);
  return clean.map((value, index) => a * value + n * noise[index]);
}

export function tensorToRgba(tensor: Float32Array, size: number): Uint8ClampedArray {
  const plane = size * size, rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    for (let channel = 0; channel < 3; channel++) rgba[i * 4 + channel] = Math.round((Math.max(-1, Math.min(1, tensor[i + plane * channel])) + 1) * 127.5);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
