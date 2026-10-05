import { useEffect, useRef, useState } from "react";

const SIZE = 32;
// Each level: which clean training whale to use, and how many pixels to corrupt.
// Noise grows about 4x per level; the last level leaves a few clean pixels showing through.
const LEVELS = [{ whale: 0, noise: 4 }, { whale: 1, noise: 16 }, { whale: 2, noise: 64 }, { whale: 3, noise: 256 }, { whale: 4, noise: 900 }];

function loadClean(whale: number): Promise<Uint8ClampedArray> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = SIZE;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(image, 0, 0);
      resolve(ctx.getImageData(0, 0, SIZE, SIZE).data);
    };
    image.onerror = reject;
    image.src = `${import.meta.env.BASE_URL}training/32/${whale}-0.png`;
  });
}

const MODIFIERS = new Set(["shift", "capslock", "control", "alt", "meta"]);
const KONAMI = ["arrowup", "arrowup", "arrowdown", "arrowdown", "arrowleft", "arrowright", "arrowleft", "arrowright", "b", "a"];

function pickNoisy(count: number): Set<number> {
  const picked = new Set<number>();
  while (picked.size < count) picked.add(Math.floor(Math.random() * SIZE * SIZE));
  return picked;
}

export function DenoiseGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [level, setLevel] = useState(0);
  const [clean, setClean] = useState<Uint8ClampedArray>();
  const [noise, setNoise] = useState<Map<number, [number, number, number]>>(new Map());
  const [remaining, setRemaining] = useState<Set<number>>(new Set());
  const [misses, setMisses] = useState(0);
  const [skipUnlocked, setSkipUnlocked] = useState(false);
  const [skipJustUnlocked, setSkipJustUnlocked] = useState(false);

  function start(levelIndex: number) {
    const { whale, noise: count } = LEVELS[levelIndex];
    setLevel(levelIndex); setMisses(0); setClean(undefined);
    loadClean(whale).then((pixels) => {
      const noisy = pickNoisy(count);
      setNoise(new Map([...noisy].map((i) => [i, [0, 1, 2].map(() => Math.floor(Math.random() * 256)) as [number, number, number]])));
      setRemaining(noisy);
      setClean(pixels);
    });
  }

  useEffect(() => { start(0); }, []);

  // Undocumented: the Konami code reveals a Skip level button, for coaches fast-forwarding a demo.
  useEffect(() => {
    const recent: string[] = [];
    const onKey = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (MODIFIERS.has(key)) return; // so Shift+B / Shift+A (or Caps Lock) still count
      recent.push(key);
      if (recent.length > KONAMI.length) recent.shift();
      if (recent.join() === KONAMI.join()) { setSkipUnlocked(true); setSkipJustUnlocked(true); window.removeEventListener("keydown", onKey); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function skip() {
    if (level < LEVELS.length - 1) start(level + 1);
    else setRemaining(new Set());
  }

  useEffect(() => {
    if (!clean) return;
    const pixels = new Uint8ClampedArray(clean);
    for (const i of remaining) pixels.set(noise.get(i)!, i * 4);
    canvasRef.current?.getContext("2d")?.putImageData(new ImageData(pixels, SIZE, SIZE), 0, 0);
  }, [clean, noise, remaining]);

  function click(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!clean || remaining.size === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * SIZE);
    const y = Math.floor((event.clientY - rect.top) / rect.height * SIZE);
    const index = y * SIZE + x;
    if (!remaining.has(index)) { setMisses((m) => m + 1); return; }
    const next = new Set(remaining); next.delete(index); setRemaining(next);
  }

  const total = LEVELS[level].noise, found = total - remaining.size, done = !!clean && remaining.size === 0;
  const last = level === LEVELS.length - 1;
  return <section className="model-section denoise-game" aria-labelledby="denoise-heading">
    <h2 id="denoise-heading">Train like a diffusion model</h2>
    <p className="section-note">Some pixels in this whale have been replaced with random noise. Click every noisy pixel to restore the original image.</p>
    <div className="generator-card">
      <div className="controls">
        <div className="timeline-heading"><span>Level {level + 1} of {LEVELS.length}</span><strong>{found} / {total} found</strong></div>
        <div className="status"><i className={`dot ${done ? "ready" : "playing"}`} />{done ? "Whale restored" : `${misses} miss${misses === 1 ? "" : "es"}`}</div>
        {done && <p className="denoise-done"><span aria-hidden>✓</span> {last ? "All levels done." : `Restored with ${misses} miss${misses === 1 ? "" : "es"}.`}</p>}
        <div className="generator-actions denoise-actions">
          {done && !last && <button className="run" onClick={() => start(level + 1)}>Next level <span>▶</span></button>}
          <button className="secondary" onClick={() => start(level)}>Reset level</button>
          <button className="secondary" onClick={() => start(0)}>Start again</button>
          {skipUnlocked && !done && <button className={`secondary denoise-skip${skipJustUnlocked ? " revealed" : ""}`} onClick={skip}
            onAnimationEnd={(event) => { if (event.animationName === "skip-reveal") setSkipJustUnlocked(false); }}>Skip level</button>}
        </div>
        <p className="denoise-disclaimer"><strong>Simplified example</strong>Real diffusion models solve a considerably harder problem.</p>
      </div>
      <div className="viewer">
        <div className="canvas-shell">
          <canvas ref={canvasRef} width={SIZE} height={SIZE} className="denoise-canvas" onClick={click}
            aria-label={`Noisy whale, ${remaining.size} noisy pixels left`} />
          {done && <span className="denoise-tick" aria-hidden>✓</span>}
        </div>
      </div>
    </div>
  </section>;
}
