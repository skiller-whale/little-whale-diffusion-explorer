import { describe, expect, test } from "bun:test";
import { ddimStep, gaussianNoise, inferenceTimesteps, mulberry32, tensorToRgba, timestepEmbedding } from "./diffusion";

describe("diffusion primitives", () => {
  test("seeded random values and Gaussian noise are repeatable", () => {
    expect(mulberry32(42)()).toBe(mulberry32(42)());
    expect([...gaussianNoise(8, 99)]).toEqual([...gaussianNoise(8, 99)]);
  });
  test("inference schedule runs from noise to clean", () => {
    expect(inferenceTimesteps(10)[0]).toBe(999);
    expect(inferenceTimesteps(10).at(-1)).toBe(0);
  });
  test("embedding and pixels have the public contract", () => {
    expect(timestepEmbedding(500)).toHaveLength(160);
    const pixels = tensorToRgba(new Float32Array(3 * 4), 2);
    expect(pixels).toHaveLength(16);
    expect(pixels[3]).toBe(255);
  });
  test("final DDIM step is finite", () => {
    const output = ddimStep(new Float32Array([.5]), new Float32Array([.1]), 0, -1);
    expect(Number.isFinite(output[0])).toBe(true);
  });
});
