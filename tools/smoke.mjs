// Headless smoke test of a deployed or local site. Usage: node tools/smoke.mjs <url> <shotDir> [mods]
import puppeteer from 'puppeteer-core'; import fs from 'node:fs';
const url = process.argv[2] || 'http://localhost:8790/', dir = process.argv[3] || '/workspace/neuralese-diffusion-shots', mods = (process.argv[4] || 'image,music,video,shape3d').split(','), tag = process.argv[5] || 'local';
fs.mkdirSync(dir, { recursive: true });
const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const out = { url, checks: [], errors: [] }, ok = (name, pass, info = '') => { out.checks.push({ name, pass, info }); console.log(pass ? 'OK  ' : 'FAIL', name, info); };
for (const mod of mods) {
  const p = await b.newPage(); await p.setViewport({ width: 1280, height: 860 }); p.on('pageerror', (e) => out.errors.push(mod + ': ' + e.message)); p.on('console', (m) => m.type() === 'error' && out.errors.push(mod + ' console: ' + m.text()));
  await p.goto(url + '?mod=' + mod, { waitUntil: 'networkidle0', timeout: 120000 }); await p.waitForFunction('window.ND && window.ND.ready', { timeout: 120000 });
  await p.click('#go'); await p.waitForFunction('document.body.innerText.includes("judge:")', { timeout: 180000 });
  const txt = await p.evaluate(() => document.querySelector('main').innerText); ok(`${mod} sample`, /judge: /.test(txt), (txt.match(/\d+ steps in \d+ ms/) || [''])[0] + ' ' + (txt.match(/judge: \S+/) || [''])[0]);
  await p.screenshot({ path: `${dir}/${tag}-${mod}-SAMPLE.png` });
  for (const tab of ['LENS', 'TRAJ3D', 'ACT3D', 'ATTN', 'SURGERY', 'STEER', 'PATCH', 'SCHEDULE', 'SAE·PROBE', 'METRICS']) {
    if (mod !== 'image' && !['LENS', 'ATTN', 'TRAJ3D'].includes(tab)) continue;
    await p.evaluate((t) => [...document.querySelectorAll('#tabs button')].find((x) => x.textContent === t).click(), tab); await new Promise((r) => setTimeout(r, 2500));
    const t2 = await p.evaluate(() => document.querySelector('main').innerText); ok(`${mod} ${tab}`, !/error:/.test(t2) && t2.length > 20, t2.slice(0, 80).replace(/\n/g, ' '));
    await p.screenshot({ path: `${dir}/${tag}-${mod}-${tab.replace('·', '-')}.png` });
  }
  if (mod === 'image') {
    const par = await p.evaluate(async () => { const M = window.ND.models.image, r = M.meta.ref; const { T3 } = await import('./engine.js'); const e = M.forward(T3(1, 16, 16, Float32Array.from(r.x)), r.t, r.y); let d = 0; e.d.forEach((v, i) => (d = Math.max(d, Math.abs(v - r.eps_cond[i])))); return d; }); ok('browser parity vs PyTorch', par < 1e-4, par.toExponential(2));
    const q = await p.evaluate(async () => { const q = await import('./tsotchke/qrng.js'); const s = await q.seed32(); return s.provenance.health + ' / NIST ' + s.provenance.nistAllPass; }); ok('QRNG WASM in browser', /pass/.test(q), q);
    const sw = await p.evaluate(() => !!document.querySelector('link[rel=manifest]')); ok('PWA manifest linked', sw);
  }
  await p.close();
}
ok('no page errors', out.errors.length === 0, out.errors.slice(0, 5).join(' | '));
fs.writeFileSync(`${dir}/${tag}-smoke.json`, JSON.stringify(out, null, 1)); await b.close(); process.exit(out.checks.every((c) => c.pass) ? 0 : 1);
