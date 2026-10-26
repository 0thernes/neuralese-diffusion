// The deploy tooling for this repo (GitHub contents API through text-only connectors) cannot carry binary files.
// Binary assets (launcher icons, splash PNGs, gradle-wrapper.jar) are stored base64 in tools/binaries.json with a
// SHA-256 each and are restored here (CI runs this first). Regenerate with: node tools/restore-binaries.mjs --pack
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto'; import { execSync } from 'node:child_process';
const F = 'tools/binaries.json', sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
if (process.argv[2] === '--pack') {
  const files = execSync('git ls-files -co --exclude-standard', { encoding: 'utf8' }).split('\n').filter(Boolean).filter((f) => fs.existsSync(f) && fs.statSync(f).size > 0 && fs.readFileSync(f).includes(0));
  const out = {}; for (const f of files) { const b = fs.readFileSync(f); out[f] = { sha256: sha(b), b64: b.toString('base64') }; }
  fs.writeFileSync(F, JSON.stringify(out)); console.log('packed', files.length, 'binaries');
} else {
  const j = JSON.parse(fs.readFileSync(F, 'utf8')); let n = 0;
  for (const [f, { sha256, b64 }] of Object.entries(j)) { const b = Buffer.from(b64, 'base64'); if (sha(b) !== sha256) throw new Error('sha mismatch ' + f); fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, b); n++; }
  console.log('restored', n, 'binaries');
}
