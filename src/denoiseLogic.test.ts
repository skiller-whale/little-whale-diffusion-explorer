import { describe, expect, test } from "bun:test";
import { mulberry32 } from "./diffusion";
import { composite, konamiDetector, LEVELS, makeNoise, pickNoisy, pixelAt, SIZE } from "./denoiseLogic";

const KONAMI = ["ArrowUp", "ArrowUp", "ArrowDown", "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowRight", "b", "a"];

describe("denoise game", () => {
  test("every level corrupts the number of pixels it promises", () => {
    for (const { noise } of LEVELS) expect(pickNoisy(noise, mulberry32(noise)).size).toBe(noise);
  });

  test("asking for more pixels than exist stops at the whole image", () => {
    expect(pickNoisy(SIZE * SIZE + 50, mulberry32(1)).size).toBe(SIZE * SIZE);
  });

  test("compositing paints only the pixels still noisy, and leaves alpha alone", () => {
    const clean = new Uint8ClampedArray(SIZE * SIZE * 4).fill(10);
    const noise = makeNoise(3, mulberry32(3));
    const [first, ...rest] = noise.keys();
    const pixels = composite(clean, noise, new Set(rest));
    expect([...pixels.slice(first * 4, first * 4 + 4)]).toEqual([10, 10, 10, 10]);
    for (const i of rest) expect([...pixels.slice(i * 4, i * 4 + 4)]).toEqual([...noise.get(i)!, 10]);
  });

  test("clicks map to the right pixel, including on the very edges", () => {
    const rect = { left: 100, top: 50, width: 320, height: 320 };
    expect(pixelAt(100, 50, rect)).toBe(0);
    expect(pixelAt(115, 65, rect)).toBe(SIZE + 1);
    expect(pixelAt(420, 370, rect)).toBe(SIZE * SIZE - 1);
    expect(pixelAt(90, 380, rect)).toBe((SIZE - 1) * SIZE);
  });

  test("the Konami code is spotted, even with Shift held or Caps Lock on, and after false starts", () => {
    const matches = konamiDetector();
    const results = ["ArrowUp", "x", ...KONAMI.slice(0, 8), "Shift", "B", "A"].map(matches);
    expect(results.at(-1)).toBe(true);
    expect(results.slice(0, -1).some(Boolean)).toBe(false);
  });

  test("an almost-Konami sequence does not count", () => {
    const matches = konamiDetector();
    expect([...KONAMI.slice(0, 9), "c"].map(matches).some(Boolean)).toBe(false);
  });
});
