// Headless check of the real tier: runtime + TAESD self-test on WASM and (SwiftShader) WebGPU. Usage: node tools/smoke-real.mjs <url> <shotDir> <tag>
import puppeteer from 'puppeteer-core'; import fs from 'node:fs';
const url = process.argv[2] || 'http://localhost:8790/real/', dir = process.argv[3] || '/workspace/neuralese-diffusion-shots/v2', tag = process.argv[4] || 'local'; fs.mkdirSync(dir, { recursive: true });
const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', protocolTimeout: 600000, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const out = { url, runs: [], errors: [] };
for (const ep of (process.argv[5] || 'wasm,webgpu').split(',')) { const p = await b.newPage(); p.on('request', (r) => /jsdelivr|huggingface/.test(r.url()) && (out.net ||= []).push(ep + ' ' + r.url())); await p.setViewport({ width: 1280, height: 900 }); p.on('pageerror', (e) => out.errors.push(ep + ': ' + e.message)); p.on('console', (m) => m.type() === 'error' && out.errors.push(ep + ' console: ' + m.text()));
  await p.goto(url + (ep === 'wasm' ? '#wasm' : ''), { waitUntil: 'networkidle0', timeout: 120000 }); await p.waitForFunction('window.NDR && window.NDR.device', { timeout: 60000 });
  await p.click('#selftest'); await p.waitForFunction('window.NDR.selftest || /SHA|Error|error/.test(document.querySelector("#st").innerText)', { timeout: 300000 });
  const r = await p.evaluate(() => ({ device: NDR.device, selftest: NDR.selftest, text: document.querySelector('#st').innerText })); out.runs.push({ want: ep, ...r }); console.log(ep, JSON.stringify(r));
  await p.screenshot({ path: `${dir}/${tag}-real-selftest-${ep}.png`, fullPage: true }); await p.close(); }
fs.writeFileSync(`${dir}/${tag}-real-smoke.json`, JSON.stringify(out, null, 1)); console.log('errors', out.errors, 'net', out.net); await b.close();
