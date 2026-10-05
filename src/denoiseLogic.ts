// Pure logic for the "Train like a diffusion model" warm-up, kept free of React so it can be tested.

export const SIZE = 32;

// Each level: which clean training whale to use, and how many pixels to corrupt.
// Noise grows about 4x per level; the last level leaves a few clean pixels showing through.
export const LEVELS = [{ whale: 0, noise: 4 }, { whale: 1, noise: 16 }, { whale: 2, noise: 64 }, { whale: 3, noise: 256 }, { whale: 4, noise: 900 }];

export type Rgb = [number, number, number];

export function pickNoisy(count: number, random = Math.random): Set<number> {
  const picked = new Set<number>();
  while (picked.size < Math.min(count, SIZE * SIZE)) picked.add(Math.floor(random() * SIZE * SIZE));
  return picked;
}

// Fully random colours: a noisy pixel can land on (or near) the colour it replaced and be hard or impossible to spot.
// That is intended, so don't filter them out.
export function noiseColour(random = Math.random): Rgb {
  return [0, 1, 2].map(() => Math.floor(random() * 256)) as Rgb;
}

export function makeNoise(count: number, random = Math.random): Map<number, Rgb> {
  return new Map([...pickNoisy(count, random)].map((i) => [i, noiseColour(random)]));
}

export function composite(clean: Uint8ClampedArray, noise: Map<number, Rgb>, remaining: Set<number>): Uint8ClampedArray<ArrayBuffer> {
  const pixels = new Uint8ClampedArray(clean);
  for (const i of remaining) pixels.set(noise.get(i)!, i * 4);
  return pixels;
}

// Which pixel a click landed on, given the canvas's on-screen box. Clamped, so a click on the very edge still counts.
export function pixelAt(clientX: number, clientY: number, rect: { left: number; top: number; width: number; height: number }): number {
  const clamp = (v: number) => Math.min(SIZE - 1, Math.max(0, Math.floor(v)));
  return clamp((clientY - rect.top) / rect.height * SIZE) * SIZE + clamp((clientX - rect.left) / rect.width * SIZE);
}

const MODIFIERS = new Set(["shift", "capslock", "control", "alt", "meta"]);
const KONAMI = ["arrowup", "arrowup", "arrowdown", "arrowdown", "arrowleft", "arrowright", "arrowleft", "arrowright", "b", "a"];

// Feed it every key press; it returns true once the most recent presses spell the Konami code.
export function konamiDetector(): (key: string) => boolean {
  const recent: string[] = [];
  return (key) => {
    key = key.toLowerCase();
    if (MODIFIERS.has(key)) return false; // so Shift+B / Shift+A (or Caps Lock) still count
    recent.push(key);
    if (recent.length > KONAMI.length) recent.shift();
    return recent.join() === KONAMI.join();
  };
}
