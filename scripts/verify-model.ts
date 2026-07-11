import manifest from "../public/models/manifest.json";

for (const entry of manifest.models) {
  const model = Bun.file(`public/models/${entry.model}`);
  if (!(await model.exists())) throw new Error(`Missing model: ${entry.model}`);
  if (model.size > 30 * 1024 * 1024) throw new Error(`Model exceeds 30 MB: ${entry.model}`);
  const hash = new Bun.CryptoHasher("sha256").update(await model.arrayBuffer()).digest("hex");
  if (hash !== entry.sha256) throw new Error(`Model checksum mismatch for ${entry.model}: ${hash}`);
  console.log(`${entry.imageSize}×${entry.imageSize} model verified: ${(model.size / 1024 ** 2).toFixed(1)} MB, sha256 ${hash.slice(0, 12)}…`);
}
