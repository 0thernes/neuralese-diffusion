import fs from 'node:fs'; const mods = ['image', 'music', 'video', 'shape3d'], results = {}; let sha = null, N = null;
for (const m of mods) { const f = `site/results/${m}.json`; if (!fs.existsSync(f)) continue; const r = JSON.parse(fs.readFileSync(f, 'utf8')); results[m] = r; sha = r.prereg_sha256; N = r.N; }
fs.writeFileSync('site/results/summary.json', JSON.stringify({ prereg_sha256: sha, N, date: new Date().toISOString(), results }, null, 1)); console.log(Object.keys(results));
