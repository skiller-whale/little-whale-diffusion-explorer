export type Backend = "webgpu" | "wasm";

export type WorkerRequest =
  | { type: "initialize"; modelUrl: string }
  | { type: "generate"; id: number; seed: number; steps: number; frameDelayMs: number }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "cancel" };

export type WorkerResponse =
  | { type: "ready"; backend: Backend }
  | { type: "status"; message: string }
  | { type: "frame"; id: number; index: number; total: number; pixels: Uint8ClampedArray; elapsedMs: number }
  | { type: "paused"; id: number }
  | { type: "complete"; id: number; elapsedMs: number }
  | { type: "cancelled"; id: number }
  | { type: "error"; message: string };
