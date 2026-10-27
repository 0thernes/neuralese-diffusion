// v3 check: LAB node importance runs in the Web Worker while the page stays responsive. Usage: node tools/labw-check.mjs (serve site/ on :8811 first)
import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', protocolTimeout: 1200000, args: ['--no-sandbox'] });
const p = await b.newPage(); const errs=[]; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => m.type()==='error' && errs.push(m.text()));
await p.goto('http://localhost:8811/?mod=image2', { waitUntil: 'networkidle0', timeout: 180000 }); await p.waitForFunction('window.ND && window.ND.ready', { timeout: 180000 });
await p.evaluate(() => [...document.querySelectorAll('button,a,[data-tab]')].find(x => x.textContent.trim() === 'LAB').click());
await p.waitForFunction(() => [...document.querySelectorAll('button')].some(x => x.textContent.startsWith('node importance')), { timeout: 60000 });
await p.evaluate(() => [...document.querySelectorAll('button')].find(x => x.textContent.startsWith('node importance')).click());
const t0 = Date.now(); let maxLag = 0;
while (true) { const t = Date.now(); const txt = await p.evaluate(() => [...document.querySelectorAll('pre')].map(x => x.textContent).join('\n')); maxLag = Math.max(maxLag, Date.now() - t);
  if (/node importance.*\(\d+\.\d s/.test(txt) || /failed:/.test(txt)) { console.log(txt.slice(0, 600)); break; }
  if (Date.now() - t0 > 900000) { console.log('timeout'); break; } await new Promise(r => setTimeout(r, 2000)); }
console.log('max evaluate latency ms', maxLag, 'errors', errs); await b.close();
