export const IMAGE_SIZE = 64;
export const CHANNELS = 3;
export const TRAINING_STEPS = 1000;
export const TIME_EMBEDDING_SIZE = 192;
export const ORCA_ARCHETYPES = 16;

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
  const f0 = Math.cos((offset / (1 + offset)) * Math.PI * 0.5) ** 2;
  for (let t = 0; t < steps; t++) {
    const x = (t + 1) / steps;
    values[t] = Math.max(Math.cos(((x + offset) / (1 + offset)) * Math.PI * 0.5) ** 2 / f0, 1e-7);
  }
  return values;
}

export function inferenceTimesteps(count: number, trainingSteps = TRAINING_STEPS): number[] {
  if (count < 2) throw new Error("At least two denoising steps are required");
  return Array.from({ length: count }, (_, i) =>
    Math.round((trainingSteps - 1) * (1 - i / (count - 1))),
  );
}

export function timestepEmbedding(timestep: number, size = TIME_EMBEDDING_SIZE): Float32Array {
  const half = size / 2;
  const embedding = new Float32Array(size);
  for (let i = 0; i < half; i++) {
    const frequency = Math.exp((-Math.log(10000) * i) / Math.max(half - 1, 1));
    embedding[i] = Math.cos(timestep * frequency);
    embedding[i + half] = Math.sin(timestep * frequency);
  }
  return embedding;
}

export function seedConditioning(seed: number): Float32Array {
  const conditioning = new Float32Array(ORCA_ARCHETYPES);
  conditioning[(seed >>> 0) % ORCA_ARCHETYPES] = 1;
  return conditioning;
}

export function ddimStep(
  sample: Float32Array,
  predictedVelocity: Float32Array,
  timestep: number,
  previousTimestep: number,
  alphas = cosineAlphaCumprod(),
): Float32Array {
  const alpha = alphas[timestep];
  const previousAlpha = previousTimestep >= 0 ? alphas[previousTimestep] : 1;
  const sqrtAlpha = Math.sqrt(alpha);
  const sqrtOneMinusAlpha = Math.sqrt(Math.max(1 - alpha, 0));
  const sqrtPreviousAlpha = Math.sqrt(previousAlpha);
  const sqrtPreviousOneMinusAlpha = Math.sqrt(Math.max(1 - previousAlpha, 0));
  const output = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++) {
    const clean = Math.max(-1, Math.min(1, sqrtAlpha * sample[i] - sqrtOneMinusAlpha * predictedVelocity[i]));
    const noise = sqrtOneMinusAlpha * sample[i] + sqrtAlpha * predictedVelocity[i];
    output[i] = sqrtPreviousAlpha * clean + sqrtPreviousOneMinusAlpha * noise;
  }
  return output;
}

export function tensorToRgba(tensor: Float32Array, size = IMAGE_SIZE): Uint8ClampedArray {
  const plane = size * size;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    rgba[i * 4] = Math.round((Math.max(-1, Math.min(1, tensor[i])) + 1) * 127.5);
    rgba[i * 4 + 1] = Math.round((Math.max(-1, Math.min(1, tensor[i + plane])) + 1) * 127.5);
    rgba[i * 4 + 2] = Math.round((Math.max(-1, Math.min(1, tensor[i + plane * 2])) + 1) * 127.5);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
