import { useCallback, useEffect, useRef, useState } from "react";
import { gaussianNoise, MODELS, tensorToRgba, type ModelConfig } from "./diffusion";
import type { WorkerRequest, WorkerResponse } from "./protocol";
import { DenoiseGame } from "./DenoiseGame";

type Phase = "loading" | "generating" | "ready" | "playing" | "error";
const TRAINING_TIMESTEPS = [0, 70, 180, 330, 500, 680, 840, 999];

function randomSeed() { return crypto.getRandomValues(new Uint32Array(1))[0]; }

function DiffusionGenerator({ config }: { config: ModelConfig }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | undefined>(undefined);
  const runIdRef = useRef(0);
  const playbackRef = useRef<number | undefined>(undefined);
  const preserveFinalFrameRef = useRef(false);
  const holdFinalFrameRef = useRef(false);
  const [phase, setPhase] = useState<Phase>("loading");
  const [status, setStatus] = useState("Loading model");
  const [seed, setSeed] = useState(() => randomSeed());
  const [steps, setSteps] = useState(config.defaultSteps);
  const [stepDraft, setStepDraft] = useState(config.defaultSteps);
  const [frames, setFrames] = useState<Uint8ClampedArray[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [savedFrames, setSavedFrames] = useState<{ id: number; pixels: Uint8ClampedArray }[]>([]);

  const draw = useCallback((pixels: Uint8ClampedArray) => {
    canvasRef.current?.getContext("2d")?.putImageData(new ImageData(new Uint8ClampedArray(pixels), config.imageSize, config.imageSize), 0, 0);
  }, [config.imageSize]);

  useEffect(() => {
    const initial = tensorToRgba(gaussianNoise(3 * config.imageSize ** 2, seed), config.imageSize);
    draw(initial);
    const worker = new Worker(new URL("./inference.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === "status") setStatus(data.message);
      if (data.type === "ready") { setModelReady(true); setStatus("Generating frames"); }
      if (data.type === "frame" && data.id === runIdRef.current) {
        const copy = new Uint8ClampedArray(data.pixels);
        setFrames((current) => { const next = current.slice(); next[data.index] = copy; return next; });
        if (data.index === 0 && !holdFinalFrameRef.current) draw(copy);
      }
      if (data.type === "complete" && data.id === runIdRef.current) { holdFinalFrameRef.current = false; setPhase("ready"); setStatus("Ready"); }
      if (data.type === "error") { holdFinalFrameRef.current = false; setPhase("error"); setStatus(data.message); }
    };
    worker.postMessage({ type: "initialize", config } satisfies WorkerRequest);
    return () => { worker.terminate(); if (playbackRef.current) clearInterval(playbackRef.current); };
  }, [config, draw]);

  useEffect(() => {
    if (!modelReady) return;
    const timeout = window.setTimeout(() => {
      if (playbackRef.current) clearInterval(playbackRef.current);
      workerRef.current?.postMessage({ type: "cancel" } satisfies WorkerRequest);
      const id = ++runIdRef.current;
      const preserveFinalFrame = preserveFinalFrameRef.current;
      holdFinalFrameRef.current = preserveFinalFrame;
      setFrames([]); setFrameIndex(preserveFinalFrame ? steps : 0); setPhase("generating"); setStatus("Generating frames");
      preserveFinalFrameRef.current = false;
      workerRef.current?.postMessage({ type: "generate", id, seed, steps } satisfies WorkerRequest);
    }, 160);
    return () => clearTimeout(timeout);
  }, [modelReady, seed, steps]);

  useEffect(() => { if (frames[frameIndex]) draw(frames[frameIndex]); }, [draw, frameIndex, frames]);

  function play() {
    if (phase !== "ready" || frames.length < steps + 1) return;
    if (playbackRef.current) clearInterval(playbackRef.current);
    setFrameIndex(0); setPhase("playing"); setStatus("Playing");
    let index = 0;
    const delay = matchMedia("(prefers-reduced-motion: reduce)").matches ? 10 : 90;
    playbackRef.current = window.setInterval(() => {
      index++;
      setFrameIndex(index);
      if (index >= steps) {
        clearInterval(playbackRef.current); playbackRef.current = undefined;
        setPhase("ready"); setStatus("Ready");
      }
    }, delay);
  }

  function scrub(index: number) {
    if (playbackRef.current) { clearInterval(playbackRef.current); playbackRef.current = undefined; setPhase("ready"); setStatus("Ready"); }
    setFrameIndex(index);
  }

  function commitSteps(value: number) {
    preserveFinalFrameRef.current = frameIndex === steps;
    setStepDraft(value);
    setSteps(value);
  }

  function changeSeed(value: number, preserveIfFinal = false) {
    preserveFinalFrameRef.current = preserveIfFinal && frameIndex === steps;
    setSeed(value >>> 0);
  }

  function saveFinalFrame() {
    const finalFrame = frames[steps];
    if (!finalFrame) return;
    setSavedFrames((current) => [...current, { id: Date.now() + current.length, pixels: new Uint8ClampedArray(finalFrame) }]);
  }

  return <section className="model-section" aria-labelledby={`${config.id}-heading`}>
    <h2 id={`${config.id}-heading`}>The resulting diffusion generator</h2>
    <div className="generator-card">
      <div className="controls">
        <div className="settings">
          <label>Seed<input type="number" min="0" max="4294967295" value={seed} onChange={(event) => changeSeed(Number(event.target.value))} /></label>
          <button className="secondary" onClick={() => changeSeed(randomSeed(), true)}>Randomise</button>
          <label className="range-setting"><span>Steps <strong>{stepDraft}</strong></span><input type="range" min="1" max="100" value={stepDraft}
            onChange={(event) => setStepDraft(Number(event.target.value))}
            onPointerUp={(event) => commitSteps(Number(event.currentTarget.value))}
            onKeyUp={(event) => commitSteps(Number(event.currentTarget.value))}
            onBlur={(event) => commitSteps(Number(event.currentTarget.value))} /></label>
        </div>
        <div className="generator-actions">
          <button className="run" onClick={play} disabled={phase !== "ready" || frames.length < steps + 1}>Play <span>▶</span></button>
          <button className="save" onClick={saveFinalFrame} disabled={phase !== "ready" || !frames[steps]}>Save</button>
        </div>
        <div className="timeline-heading"><span>Denoising timeline</span><strong>{frameIndex} / {steps}</strong></div>
        <input className="timeline" type="range" min="0" max={steps} value={Math.min(frameIndex, steps)} disabled={phase === "loading" || phase === "generating" || !frames.length}
          onChange={(event) => scrub(Math.min(Number(event.target.value), frames.length - 1))}
          style={{ "--progress": `${frameIndex / steps * 100}%` } as React.CSSProperties} aria-label={`${config.label} denoising timeline`} />
        <div className="status"><i className={`dot ${phase}`} />{phase === "error" ? "Model error" : status}</div>
        {phase === "error" && <p className="error"><code>{status}</code></p>}
      </div>
      <div className="viewer">
        <div className="canvas-shell">
          <canvas ref={canvasRef} width={config.imageSize} height={config.imageSize} aria-label={`${config.label} diffusion at frame ${frameIndex}`} />
          <span className="corner">{config.label}</span>
        </div>
      </div>
    </div>
    {savedFrames.length > 0 && <div className={`saved-gallery saved-gallery-${config.imageSize}`} aria-label={`Saved ${config.label} images`}>
      <div className="gallery-heading">Saved images <span>{savedFrames.length}</span></div>
      <div className="gallery-images">{savedFrames.map((saved, index) => <PixelCell key={saved.id} pixels={saved.pixels} size={config.imageSize} label={`Saved whale ${index + 1}`} />)}</div>
    </div>}
  </section>;
}

function PixelCell({ pixels, size, label }: { pixels: Uint8ClampedArray; size: number; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => { ref.current?.getContext("2d")?.putImageData(new ImageData(new Uint8ClampedArray(pixels), size, size), 0, 0); }, [pixels, size]);
  return <canvas ref={ref} width={size} height={size} aria-label={label} />;
}

function TrainingIllustrator({ config }: { config: ModelConfig }) {
  return <section className="training-section" aria-labelledby={`${config.id}-training`}>
    <h2 id={`${config.id}-training`}>Training the model</h2>
    <p className="section-note">We procedurally generate illustrations of whales, add varying amounts of noise, and train the model to predict the information needed to recover the original image.</p>
    <div className="table-scroll"><table><thead><tr>{TRAINING_TIMESTEPS.map((t, index) => <th key={t}>{index === 0 ? "Clean" : `t ${t}`}</th>)}</tr></thead>
      <tbody>{Array.from({ length: 5 }, (_, row) => <tr key={row}>{TRAINING_TIMESTEPS.map((timestep, column) => <td key={column}><img className="training-pixel" src={`${import.meta.env.BASE_URL}training/32/${row}-${column}.png`} alt={`Training example ${row + 1}, ${column === 0 ? "clean" : `noise timestep ${timestep}`}`} /></td>)}</tr>)}</tbody></table></div>
  </section>;
}

type FluxVariant = { id: string; label: string; prompt: string; seed: number; steps: number; frameCount: number };
type FluxManifest = { model: string; size: number; rows: { title: string; variants: FluxVariant[] }[] };

function fluxFrameUrl(variantId: string, index: number) {
  return `${import.meta.env.BASE_URL}flux/${variantId}/${String(index).padStart(3, "0")}.webp`;
}

function loadFluxFrames(entry: FluxVariant) {
  return Promise.all(Array.from({ length: entry.frameCount }, (_, index) => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = resolve; image.onerror = reject;
    image.src = fluxFrameUrl(entry.id, index);
  })));
}

function FluxShowcase() {
  const playbackRef = useRef<number | undefined>(undefined);
  const selectTokenRef = useRef(0);
  const loadedRef = useRef<Set<string>>(new Set());
  const [manifest, setManifest] = useState<FluxManifest | undefined>(undefined);
  const [variant, setVariant] = useState<FluxVariant | undefined>(undefined);
  const [phase, setPhase] = useState<Phase>("loading");
  const [frameIndex, setFrameIndex] = useState(0);

  async function select(next: FluxVariant) {
    const token = ++selectTokenRef.current;
    if (playbackRef.current) { clearInterval(playbackRef.current); playbackRef.current = undefined; }
    setVariant(next);
    if (!loadedRef.current.has(next.id)) {
      setPhase("loading");
      try {
        await loadFluxFrames(next);
        loadedRef.current.add(next.id);
      } catch {
        if (token === selectTokenRef.current) setPhase("error");
        return;
      }
    }
    if (token !== selectTokenRef.current) return;
    setFrameIndex(next.frameCount - 1);
    setPhase("ready");
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}flux/manifest.json`);
        if (!response.ok) throw new Error(`manifest ${response.status}`);
        const data: FluxManifest = await response.json();
        if (cancelled) return;
        setManifest(data);
        const variants = data.rows.flatMap((row) => row.variants);
        await select(variants.find((entry) => entry.steps === 20) ?? variants[0]);
        for (const entry of variants) {
          if (cancelled) return;
          if (loadedRef.current.has(entry.id)) continue;
          try {
            await loadFluxFrames(entry);
            loadedRef.current.add(entry.id);
          } catch { /* a click on this variant retries and surfaces the error */ }
        }
      } catch {
        if (!cancelled) setPhase("error");
      }
    })();
    return () => { cancelled = true; if (playbackRef.current) clearInterval(playbackRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function play() {
    if (phase !== "ready" || !variant) return;
    setPhase("playing");
    const last = variant.frameCount - 1;
    const delay = matchMedia("(prefers-reduced-motion: reduce)").matches ? 10 : 240;
    setFrameIndex(0);
    let index = 0;
    playbackRef.current = window.setInterval(() => {
      index++;
      setFrameIndex(index);
      if (index >= last) {
        clearInterval(playbackRef.current); playbackRef.current = undefined;
        setPhase("ready");
      }
    }, delay);
  }

  function scrub(value: number) {
    if (playbackRef.current) { clearInterval(playbackRef.current); playbackRef.current = undefined; setPhase("ready"); }
    setFrameIndex(value);
  }

  const last = variant ? variant.frameCount - 1 : 1;
  return <section className="model-section" aria-labelledby="flux-heading">
    <h2 id="flux-heading">Modern image model (FLUX.1 Schnell, 2024)</h2>
    <div className="generator-card">
      <div className="controls">
        {manifest && <div className="flux-thumb-groups">
          {manifest.rows.map((row) => <div key={row.title}>
            <div className="gallery-heading">{row.title}</div>
            <div className="flux-thumbs">{row.variants.map((entry) => <button key={entry.id} type="button"
              className={`thumb${variant?.id === entry.id ? " selected" : ""}`} title={entry.prompt}
              aria-pressed={variant?.id === entry.id} onClick={() => select(entry)}
              aria-label={`${entry.label} — ${entry.prompt}`}>
              <img src={fluxFrameUrl(entry.id, entry.frameCount - 1)} loading="lazy" alt="" />
              <span>{entry.label}</span>
            </button>)}</div>
          </div>)}
        </div>}
        <div className="generator-actions flux-actions">
          <button className="run" onClick={play} disabled={phase !== "ready"}>Play <span>▶</span></button>
        </div>
        <div className="timeline-heading"><span>Denoising timeline</span><strong>{frameIndex} / {last}</strong></div>
        <input className="timeline" type="range" min="0" max={last} value={Math.min(frameIndex, last)} disabled={phase === "loading" || phase === "error"}
          onChange={(event) => scrub(Number(event.target.value))}
          style={{ "--progress": `${frameIndex / last * 100}%` } as React.CSSProperties} aria-label="FLUX denoising timeline" />
        <div className="status"><i className={`dot ${phase}`} />{phase === "loading" ? "Loading frames" : phase === "error" ? "Frames unavailable" : phase === "playing" ? "Playing" : "Ready"}</div>
        {phase === "error" && <p className="error">Pre-rendered frames are missing — run <code>training/generate_flux_frames.py</code> to create them.</p>}
        {manifest && variant && <p className="flux-caption">{manifest.model} · “{variant.prompt}” · seed {variant.seed} · {variant.steps} step{variant.steps > 1 ? "s" : ""} ·
          frames pre-rendered offline</p>}
      </div>
      <div className="viewer">
        <div className="canvas-shell" role="img" aria-label={variant ? `FLUX denoising after ${frameIndex} of ${last} steps` : "FLUX denoising"}>
          {variant && phase !== "loading" && phase !== "error" &&
            <img className="flux-frame" src={fluxFrameUrl(variant.id, Math.min(frameIndex, last))} alt="" aria-hidden />}
          <span className="corner">FLUX.1 schnell</span>
        </div>
      </div>
    </div>
  </section>;
}

export function App() {
  return <main>
    <header><h1>Little Whale Diffusion Explorer</h1></header>
    <DenoiseGame />
    <TrainingIllustrator config={MODELS[0]} />
    <DiffusionGenerator config={MODELS[0]} />
    <FluxShowcase />
  </main>;
}
