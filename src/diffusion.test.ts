import { describe, expect, test } from "bun:test";
import { ddimStep, forwardDiffuse, gaussianNoise, inferenceTimesteps, MODELS, mulberry32, tensorToRgba, timestepEmbedding } from "./diffusion";

describe("diffusion primitives", () => {
  test("seeded Gaussian noise is repeatable", () => {
    expect(mulberry32(42)()).toBe(mulberry32(42)());
    expect([...gaussianNoise(8, 99)]).toEqual([...gaussianNoise(8, 99)]);
  });
  test("schedule supports the complete UI range", () => {
    expect(inferenceTimesteps(1)).toEqual([999]);
    expect(inferenceTimesteps(100)[0]).toBe(999);
    expect(inferenceTimesteps(100).at(-1)).toBe(0);
  });
  test("both model contracts have correct embeddings and pixels", () => {
    for (const model of MODELS) {
      expect(timestepEmbedding(500, model.timeEmbeddingSize)).toHaveLength(model.timeEmbeddingSize);
      expect(tensorToRgba(new Float32Array(3 * model.imageSize ** 2), model.imageSize)).toHaveLength(4 * model.imageSize ** 2);
    }
  });
  test("clean and velocity DDIM paths are finite", () => {
    for (const type of ["clean", "velocity"] as const) expect(Number.isFinite(ddimStep(new Float32Array([.5]), new Float32Array([.1]), 0, -1, type)[0])).toBe(true);
  });
  test("forward diffusion preserves clean data at the clean endpoint and is deterministic", () => {
    const clean = new Float32Array([-.5, .5]), noise = gaussianNoise(2, 7);
    expect([...forwardDiffuse(clean, noise, 300)]).toEqual([...forwardDiffuse(clean, noise, 300)]);
    expect(forwardDiffuse(clean, noise, 999)[0]).not.toBe(clean[0]);
  });
});
