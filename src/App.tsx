import { useCallback, useEffect, useRef, useState } from "react";
import { gaussianNoise, IMAGE_SIZE, tensorToRgba } from "./diffusion";
import type { Backend, WorkerRequest, WorkerResponse } from "./protocol";

type Phase = "loading" | "idle" | "running" | "paused" | "complete" | "error";
const STEP_OPTIONS = [10, 20, 30, 50];

function randomSeed() {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | undefined>(undefined);
  const runIdRef = useRef(0);
  const [phase, setPhase] = useState<Phase>("loading");
  const [status, setStatus] = useState("Preparing the model…");
  const [backend, setBackend] = useState<Backend>();
  const [seed, setSeed] = useState(() => randomSeed());
  const [steps, setSteps] = useState(20);
  const [frames, setFrames] = useState<Uint8ClampedArray[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  const draw = useCallback((pixels: Uint8ClampedArray) => {
    const context = canvasRef.current?.getContext("2d");
    if (context) context.putImageData(new ImageData(new Uint8ClampedArray(pixels), IMAGE_SIZE, IMAGE_SIZE), 0, 0);
  }, []);

  const resetNoise = useCallback((value: number) => {
    const noise = tensorToRgba(gaussianNoise(3 * IMAGE_SIZE * IMAGE_SIZE, value));
    setFrames([noise]);
    setFrameIndex(0);
    draw(noise);
  }, [draw]);

  useEffect(() => {
    resetNoise(seed);
    const worker = new Worker(new URL("./inference.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === "status") setStatus(data.message);
      if (data.type === "ready") { setBackend(data.backend); setPhase("idle"); setStatus("Ready to denoise"); }
      if (data.type === "frame" && data.id === runIdRef.current) {
        const copy = new Uint8ClampedArray(data.pixels);
        setFrames((current) => [...current.slice(0, data.index), copy]);
        setFrameIndex(data.index);
        setElapsed(data.elapsedMs);
        draw(copy);
      }
      if (data.type === "paused" && data.id === runIdRef.current) setPhase("paused");
      if (data.type === "complete" && data.id === runIdRef.current) { setPhase("complete"); setElapsed(data.elapsedMs); setStatus("A whale emerged"); }
      if (data.type === "cancelled" && data.id === runIdRef.current) setPhase("idle");
      if (data.type === "error") { setPhase("error"); setStatus(data.message); }
    };
    worker.postMessage({ type: "initialize", modelUrl: "/models/whale-ddpm.onnx" } satisfies WorkerRequest);
    return () => worker.terminate();
  }, [draw, resetNoise]);

  useEffect(() => { if (frames[frameIndex]) draw(frames[frameIndex]); }, [draw, frameIndex, frames]);

  function generate() {
    const id = ++runIdRef.current;
    resetNoise(seed);
    setElapsed(0);
    setPhase("running");
    setStatus("Finding the clean image, one step at a time…");
    const frameDelayMs = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 180;
    workerRef.current?.postMessage({ type: "generate", id, seed, steps, frameDelayMs } satisfies WorkerRequest);
  }

  function togglePause() {
    if (phase === "paused") { setPhase("running"); workerRef.current?.postMessage({ type: "resume" } satisfies WorkerRequest); }
    else workerRef.current?.postMessage({ type: "pause" } satisfies WorkerRequest);
  }

  function cancel() {
    workerRef.current?.postMessage({ type: "cancel" } satisfies WorkerRequest);
    runIdRef.current += 1;
    resetNoise(seed);
    setElapsed(0);
    setPhase("idle");
    setStatus("Cancelled — ready to start again");
  }

  function chooseSeed(value: number) {
    if (phase === "running" || phase === "paused") workerRef.current?.postMessage({ type: "cancel" } satisfies WorkerRequest);
    const normalized = value >>> 0;
    setSeed(normalized);
    resetNoise(normalized);
    if (phase !== "loading" && phase !== "error") setPhase("idle");
  }

  const busy = phase === "running" || phase === "paused";
  const progress = Math.round((frameIndex / steps) * 100);

  return <main>
    <header className="hero">
      <h1>Little Whale Diffusion</h1>
      <p>Start with static. Let a neural network predict what doesn’t belong. Repeat until something with a tail swims into view.</p>
    </header>

    <section className="lab" aria-label="Diffusion playground">
      <div className="viewer">
        <div className="canvas-shell">
          <canvas ref={canvasRef} width={IMAGE_SIZE} height={IMAGE_SIZE} aria-label={`Diffusion image at step ${frameIndex}`} />
          <div className="scanline" />
          <span className="corner corner-tl">x<sub>{frameIndex === 0 ? "T" : frameIndex}</sub></span>
          <span className="corner corner-br">64 × 64</span>
        </div>
        <div className="readout">
          <span><i className={`dot ${phase}`} />{status}</span>
          <span>{backend?.toUpperCase() ?? "—"}</span>
        </div>
      </div>

      <div className="controls">
        <div className="progress-heading"><span>Denoising timeline</span><strong>{frameIndex} / {steps}</strong></div>
        <input className="timeline" type="range" min="0" max={Math.max(steps, frames.length - 1)} value={frameIndex}
          disabled={frames.length < 2} onChange={(event) => setFrameIndex(Math.min(Number(event.target.value), frames.length - 1))}
          style={{ "--progress": `${progress}%` } as React.CSSProperties} aria-label="Denoising step" />
        <div className="transport">
          <button className="icon-button" onClick={() => setFrameIndex(Math.max(0, frameIndex - 1))} disabled={frameIndex === 0} aria-label="Previous frame">←</button>
          <button className="icon-button" onClick={togglePause} disabled={!busy} aria-label={phase === "paused" ? "Resume" : "Pause"}>{phase === "paused" ? "▶" : "Ⅱ"}</button>
          <button className="icon-button" onClick={() => setFrameIndex(Math.min(frames.length - 1, frameIndex + 1))} disabled={frameIndex >= frames.length - 1} aria-label="Next frame">→</button>
          <span className="time">{elapsed ? `${(elapsed / 1000).toFixed(1)}s` : "—"}</span>
        </div>

        <div className="settings">
          <label>Seed<input type="number" min="0" max="4294967295" value={seed} disabled={busy} onChange={(event) => chooseSeed(Number(event.target.value))} /></label>
          <button className="shuffle" disabled={busy} onClick={() => chooseSeed(randomSeed())}>Randomise</button>
          <label>Steps<select value={steps} disabled={busy} onChange={(event) => setSteps(Number(event.target.value))}>{STEP_OPTIONS.map((value) => <option key={value}>{value}</option>)}</select></label>
        </div>

        <button className="generate" onClick={busy ? cancel : generate} disabled={phase === "loading" || phase === "error"}>
          <span>{busy ? "Cancel denoising" : phase === "complete" ? "Make another whale" : "Denoise a whale"}</span><span>{busy ? "×" : "↗"}</span>
        </button>
        {phase === "error" && <p className="error">The model could not load. Run the training/export command or restore the shipped model, then refresh.<br /><code>{status}</code></p>}
      </div>
    </section>

    <section className="how">
      <article><span>01</span><h2>Noise</h2><p>A seed selects an orca archetype and a repeatable starting cloud.</p></article>
      <article><span>02</span><h2>Predict</h2><p>The U-Net predicts a stable blend of image and noise at each timestep.</p></article>
      <article><span>03</span><h2>Denoise</h2><p>DDIM removes that prediction, gradually revealing a whale.</p></article>
    </section>
    <footer>Inference stays on your device. No prompts, pixels, or whales leave this page.</footer>
  </main>;
}
