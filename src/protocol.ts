import type { ModelConfig } from "./diffusion";

export type WorkerRequest =
  | { type: "initialize"; config: ModelConfig }
  | { type: "generate"; id: number; seed: number; steps: number }
  | { type: "cancel" };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "status"; message: string }
  | { type: "frame"; id: number; index: number; total: number; pixels: Uint8ClampedArray; elapsedMs: number }
  | { type: "complete"; id: number; elapsedMs: number }
  | { type: "cancelled"; id: number }
  | { type: "error"; message: string };
