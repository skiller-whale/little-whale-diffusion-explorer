import { useEffect, useRef, useState } from "react";
import { composite, konamiDetector, LEVELS, makeNoise, pixelAt, type Rgb, SIZE } from "./denoiseLogic";

const cleanWhales = new Map<number, Promise<Uint8ClampedArray>>();

function loadClean(whale: number): Promise<Uint8ClampedArray> {
  if (!cleanWhales.has(whale)) {
    const loading = new Promise<Uint8ClampedArray>((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = SIZE;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(image, 0, 0);
        resolve(ctx.getImageData(0, 0, SIZE, SIZE).data);
      };
      image.onerror = () => reject(new Error(`Could not load training whale ${whale}`));
      image.src = `${import.meta.env.BASE_URL}training/32/${whale}-0.png`;
    });
    loading.catch(() => cleanWhales.delete(whale)); // let a retry fetch it again
    cleanWhales.set(whale, loading);
  }
  return cleanWhales.get(whale)!;
}

const ARROWS: Record<string, [number, number]> = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
const misses = (n: number) => `${n} miss${n === 1 ? "" : "es"}`;

export function DenoiseGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const loadId = useRef(0);
  const [level, setLevel] = useState(0);
  const [clean, setClean] = useState<Uint8ClampedArray>();
  const [failed, setFailed] = useState(false);
  const [noise, setNoise] = useState<Map<number, Rgb>>(new Map());
  const [remaining, setRemaining] = useState<Set<number>>(new Set());
  const [missCount, setMissCount] = useState(0);
  const [cursor, setCursor] = useState(SIZE * SIZE / 2 + SIZE / 2);
  const [showCursor, setShowCursor] = useState(false);
  const [skipUnlocked, setSkipUnlocked] = useState(false);
  const [skipJustUnlocked, setSkipJustUnlocked] = useState(false);

  function start(levelIndex: number) {
    const id = ++loadId.current; // a slower, older load must not overwrite a newer level
    setLevel(levelIndex); setMissCount(0); setClean(undefined); setFailed(false);
    loadClean(LEVELS[levelIndex].whale).then((pixels) => {
      if (id !== loadId.current) return;
      const levelNoise = makeNoise(LEVELS[levelIndex].noise);
      setNoise(levelNoise);
      setRemaining(new Set(levelNoise.keys()));
      setClean(pixels);
    }, () => { if (id === loadId.current) setFailed(true); });
  }

  useEffect(() => { start(0); }, []);

  // Deliberately unadvertised: the Konami code reveals a Skip level button, for coaches fast-forwarding a demo.
  useEffect(() => {
    const matches = konamiDetector();
    const onKey = (event: KeyboardEvent) => {
      if (!matches(event.key)) return;
      setSkipUnlocked(true); setSkipJustUnlocked(true);
      window.removeEventListener("keydown", onKey);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (clean) canvasRef.current?.getContext("2d")?.putImageData(new ImageData(composite(clean, noise, remaining), SIZE, SIZE), 0, 0);
  }, [clean, noise, remaining]);

  const done = !!clean && remaining.size === 0;

  function pick(index: number) {
    if (!clean || done) return;
    if (!remaining.has(index)) { setMissCount((m) => m + 1); return; }
    const next = new Set(remaining); next.delete(index); setRemaining(next);
  }

  function skip() {
    if (level < LEVELS.length - 1) start(level + 1);
    else setRemaining(new Set());
  }

  function onCanvasKey(event: React.KeyboardEvent<HTMLCanvasElement>) {
    const move = ARROWS[event.key];
    if (move) {
      event.preventDefault(); // arrows move the cursor instead of scrolling (the Konami listener still sees them)
      const x = Math.min(SIZE - 1, Math.max(0, cursor % SIZE + move[0]));
      const y = Math.min(SIZE - 1, Math.max(0, Math.floor(cursor / SIZE) + move[1]));
      setCursor(y * SIZE + x); setShowCursor(true);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setShowCursor(true); pick(cursor);
    }
  }

  const total = LEVELS[level].noise, found = total - remaining.size, last = level === LEVELS.length - 1;
  const status = failed ? "Couldn't load" : !clean ? "Loading" : done ? "Whale restored" : misses(missCount);
  const announcement = failed ? "The whale image couldn't be loaded."
    : !clean ? `Loading level ${level + 1}.`
    : done ? `${last ? "All levels done" : "Whale restored"} with ${misses(missCount)}.`
    : `Level ${level + 1} of ${LEVELS.length}: ${found} of ${total} noisy pixels found, ${misses(missCount)}.`;

  return <section className="model-section denoise-game" aria-labelledby="denoise-heading">
    <h2 id="denoise-heading">Train like a diffusion model</h2>
    <p className="section-note" id="denoise-instructions">Some pixels in this whale have been replaced with random noise. Click every noisy pixel to restore the original image.</p>
    <div className="generator-card">
      <div className="controls">
        <div className="timeline-heading"><span>Level {level + 1} of {LEVELS.length}</span><strong>{clean ? `${found} / ${total} found` : "–"}</strong></div>
        <div className="status"><i className={`dot ${done ? "ready" : "playing"}`} />{status}</div>
        <p className="visually-hidden" aria-live="polite">{announcement}</p>
        {failed && <p className="error denoise-error">The whale image couldn't be loaded. Check your connection, then try again.
          <button className="secondary" onClick={() => start(level)}>Try again</button></p>}
        {done && <p className="denoise-done"><span aria-hidden>✓</span> {last ? "All levels done." : `Restored with ${misses(missCount)}.`}</p>}
        <div className="generator-actions denoise-actions">
          {done && !last && <button className="run" onClick={() => start(level + 1)}>Next level <span aria-hidden>▶</span></button>}
          <button className="secondary" onClick={() => start(level)}>Reset level</button>
          <button className="secondary" onClick={() => start(0)}>Start again</button>
          {skipUnlocked && !done && <button className={`secondary denoise-skip${skipJustUnlocked ? " revealed" : ""}`} onClick={skip} disabled={!clean}
            onAnimationEnd={(event) => { if (event.animationName === "skip-reveal") setSkipJustUnlocked(false); }}>Skip level</button>}
        </div>
        <p className="denoise-disclaimer"><strong>Simplified example</strong>Real diffusion models solve a considerably harder problem.</p>
      </div>
      <div className="viewer">
        <div className="canvas-shell">
          <canvas ref={canvasRef} width={SIZE} height={SIZE} className="denoise-canvas" tabIndex={0}
            aria-label="Noisy whale. Use the arrow keys to move, and Enter to pick a pixel." aria-describedby="denoise-instructions"
            onClick={(event) => { setShowCursor(false); pick(pixelAt(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect())); }}
            onKeyDown={onCanvasKey} onBlur={() => setShowCursor(false)} />
          {showCursor && !done && <span className="denoise-cursor" aria-hidden
            style={{ left: `${cursor % SIZE / SIZE * 100}%`, top: `${Math.floor(cursor / SIZE) / SIZE * 100}%`, width: `${100 / SIZE}%`, height: `${100 / SIZE}%` }} />}
          {done && <span className="denoise-tick" aria-hidden>✓</span>}
        </div>
      </div>
    </div>
  </section>;
}
