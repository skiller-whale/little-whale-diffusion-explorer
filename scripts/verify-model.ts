import manifest from "../public/models/manifest.json";

const model = Bun.file(`public/models/${manifest.model}`);
if (!(await model.exists())) throw new Error(`Missing model: ${manifest.model}`);
if (model.size > 30 * 1024 * 1024) throw new Error(`Model exceeds 30 MB: ${(model.size / 1024 ** 2).toFixed(1)} MB`);
const hash = new Bun.CryptoHasher("sha256").update(await model.arrayBuffer()).digest("hex");
if (hash !== manifest.sha256) throw new Error(`Model checksum mismatch: ${hash}`);
console.log(`Model verified: ${(model.size / 1024 ** 2).toFixed(1)} MB, sha256 ${hash.slice(0, 12)}…`);
