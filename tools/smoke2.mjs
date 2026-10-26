// Headless smoke of the v2 self-trained modalities + LAB tab. Usage: node tools/smoke2.mjs <url> <shotDir> <tag>
import puppeteer from 'puppeteer-core'; import fs from 'node:fs';
const url = process.argv[2], dir = process.argv[3], tag = process.argv[4] || 'live'; fs.mkdirSync(dir, { recursive: true });
const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const out = { url, checks: [], errors: [] }, ok = (n, p, i = '') => { out.checks.push({ name: n, pass: p, info: i }); console.log(p ? 'OK  ' : 'FAIL', n, i); };
for (const mod of ['image2', 'audio2', 'video2', 'pc3d2']) {
  const p = await b.newPage(); await p.setViewport({ width: 1280, height: 900 }); p.on('pageerror', (e) => out.errors.push(mod + ': ' + e.message));
  await p.goto(url + '?mod=' + mod, { waitUntil: 'networkidle0', timeout: 180000 }); await p.waitForFunction('window.ND && window.ND.ready', { timeout: 180000 });
  const sel = await p.evaluate(() => document.querySelector('#mod').value); ok(`${mod} selected`, sel === mod, sel);
  await p.click('#go'); await p.waitForFunction('document.body.innerText.includes("judge:")', { timeout: 600000 });
  const txt = await p.evaluate(() => document.querySelector('main').innerText); ok(`${mod} sample`, /judge: /.test(txt), (txt.match(/\d+ steps in \d+ ms/) || [''])[0] + ' ' + (txt.match(/judge: \S+/) || [''])[0]);
  await p.screenshot({ path: `${dir}/${tag}-${mod}-SAMPLE.png` });
  for (const tab of ['LENS', 'LAB', 'METRICS']) {
    const has = await p.evaluate((t) => { const x = [...document.querySelectorAll('#tabs button')].find((x) => x.textContent === t); if (x) x.click(); return !!x; }, tab);
    await new Promise((r) => setTimeout(r, tab === 'LAB' ? 15000 : 3000));
    const t2 = await p.evaluate(() => document.querySelector('main').innerText); ok(`${mod} ${tab}`, has && !/error:/i.test(t2) && t2.length > 20, t2.slice(0, 90).replace(/\n/g, ' '));
    if (tab === 'LAB' && mod === 'image2') { const before = await p.evaluate(() => document.querySelector('main').innerText.length);
      await p.evaluate(() => [...document.querySelectorAll('main button')].find((x) => x.textContent.startsWith('node importance')).click());
      await p.waitForFunction((n) => document.querySelector('main').innerText.length > n + 40, { timeout: 600000 }, before).catch(() => {});
      const t3 = await p.evaluate(() => document.querySelector('main').innerText); ok('image2 LAB node importance ran', t3.length > before + 40, t3.slice(before, before + 120).replace(/\n/g, ' ')); }
    await p.screenshot({ path: `${dir}/${tag}-${mod}-${tab}.png`, fullPage: tab === 'LAB' });
  }
  await p.close();
}
ok('no page errors', out.errors.length === 0, out.errors.slice(0, 5).join(' | '));
fs.writeFileSync(`${dir}/${tag}-smoke2.json`, JSON.stringify(out, null, 1)); await b.close(); process.exit(out.checks.every((c) => c.pass) ? 0 : 1);
