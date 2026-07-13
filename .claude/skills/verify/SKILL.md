---
name: verify
description: Build, launch, and drive this app in a headless browser to verify changes at the browser surface.
---

# Verifying little-whale-diffusion-explorer

Vite + React app, package manager is bun.

## Launch

```bash
bun run dev   # background; prints the port (5173/5174/…)
```

The app is served under the Vite base path: `http://localhost:<port>/little-whale-diffusion-explorer/` — the bare root 404s. Same base applies to asset URLs (e.g. `/little-whale-diffusion-explorer/flux/manifest.json`).

## Drive

No Playwright in the repo. Install `playwright-core` in the scratchpad and use the cached headless shell:

```
executablePath: ~/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell
```

(`chromium-1223` holds "Google Chrome for Testing.app" under `chrome-mac-arm64/`, not `chrome-mac/`.)

Run scripts with `node` (not bun) from the scratchpad.

## Gotchas

- Status text renders uppercase via CSS — `innerText` returns "READY"/"PLAYING", so match case-insensitively.
- The FLUX section fetches `flux/manifest.json` then preloads all variant frames in the background on page load; watch `page.on("request")` for `/flux/**.webp` to observe it.
- The ONNX diffusion generators load models in a worker and take a while; the FLUX section is independent of them.

## Flows worth driving

- FLUX section: thumbnails select variants, Play animates the denoising timeline, timeline slider scrubs.
- Orca generators: seed/steps controls regenerate frames; Play/scrub/Save.
