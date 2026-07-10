import { cp, mkdir, readdir } from "node:fs/promises";
import { join } from "node:path";

const source = "node_modules/onnxruntime-web/dist";
const target = process.argv[2] ?? "dist/ort";
await mkdir(target, { recursive: true });
for (const file of await readdir(source)) {
  if (/^ort-wasm.*\.(?:wasm|mjs)$/.test(file)) await cp(join(source, file), join(target, file));
}
