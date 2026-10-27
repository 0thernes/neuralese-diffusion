// Real-tier worker: verifies + loads onnxruntime-web and the pinned ONNX graphs, runs the SD pipeline with interpretability hooks.
import { ORT, TAESD, PIPES } from './models.js?v=2';
import { ClipTokenizer } from './clip-tokenizer.js?v=2';
import { SDPipeline, ablateToken, steer, toRGBA, randn } from './sd.js?v=2';
const hex = (b) => [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, '0')).join('');
let ort, pipe, pipeKey, ep;
async function fetchVerified(f, what, post) { // Cache API keeps the verified bytes; integrity is re-checked on every load
  const cache = await caches.open('nd-real-v2'); let r = await cache.match(f.url), fromCache = !!r;
  if (!r) { r = await fetch(f.url); if (!r.ok) throw new Error(`${what}: HTTP ${r.status}`); }
  const reader = r.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; post({ stage: what, progress: got / f.bytes, fromCache }); }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  const h = hex(await crypto.subtle.digest('SHA-256', buf)); if (h !== f.sha256) { await cache.delete(f.url); throw new Error(`${what}: SHA-256 mismatch (${h})`); }
  if (!fromCache) try { await cache.put(f.url, new Response(buf)); } catch { /* quota: fine, just not cached */ }
  return buf; }
async function lib(post, want) { if (ort) return ort; const js = await fetchVerified(ORT.js, 'onnxruntime-web', post), wasm = await fetchVerified(ORT.wasm, 'ort wasm', post);
  ort = await import(URL.createObjectURL(new Blob([js], { type: 'text/javascript' }))); ort.env.wasm.wasmBinary = wasm.buffer; const glue = await fetchVerified(ORT.glue, 'ort glue', post); ort.env.wasm.wasmPaths = { mjs: URL.createObjectURL(new Blob([new TextDecoder().decode(glue).replaceAll('import.meta.url', JSON.stringify(ORT.glue.url))], { type: 'text/javascript' })), wasm: ORT.wasm.url }; ort.env.wasm.proxy = false; // after the hash check, the only edit is import.meta.url -> its real CDN URL (a blob: module has no usable base); every JS/wasm byte the runtime executes is SHA-256-verified (wasm comes from wasmBinary) ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ep = want === 'wasm' || !('gpu' in navigator) || !(await navigator.gpu.requestAdapter().catch(() => null)) ? 'wasm' : 'webgpu'; return ort; }
async function session(buf, extra = {}) { return ort.InferenceSession.create(buf, { executionProviders: [ep], ...extra }); }
async function load(key, post, want) { await lib(post, want); if (pipeKey === key) return; const P = PIPES.find((p) => p.key === key), t0 = performance.now();
  const dec = new TextDecoder(), vocab = JSON.parse(dec.decode(await fetchVerified(P.tok.vocab, 'vocab', post))), merges = dec.decode(await fetchVerified(P.tok.merges, 'merges', post));
  const s = { taesd: await session(await fetchVerified(TAESD.file, 'TAESD', post)) };
  for (const k of ['text', 'unet', 'vae']) { s[k] = await session(await fetchVerified(P.files[k], `${key}/${k}`, post), P.ortOpts); }
  pipe = new SDPipeline(ort, s, new ClipTokenizer(vocab, merges, { pad: P.cfg.pad }), { spacing: P.cfg.spacing, hidden: P.cfg.hidden }); pipeKey = key;
  post({ stage: 'ready', ep, load_ms: performance.now() - t0 }); }
const img = async (lat, which = 'taesd') => toRGBA(await pipe.decode(lat, which));
onmessage = async ({ data: m }) => { const post = (x) => postMessage({ id: m.id, ...x });
  try {
    if (m.op === 'load') return await load(m.key, post, m.ep);
    if (m.op === 'taesdtest') { // phone-tier check: verified runtime + TAESD decode of a fixed latent, compared with the box (onnxruntime-node) reference
      await lib(post, m.ep); const t0 = performance.now(), s = await session(await fetchVerified(TAESD.file, 'TAESD', post)), z = randn(16384, 0).map((v) => v * 0.8), t1 = performance.now();
      const o = await s.run({ latent_sample: new ort.Tensor('float32', z, [1, 4, 64, 64]) }), y = o.sample.data, t2 = performance.now();
      return post({ done: true, ep, load_ms: t1 - t0, decode_ms: t2 - t1, out: Array.from(y.subarray(0, 8)), mean: y.reduce((a, b) => a + b, 0) / y.length, image: toRGBA(y) }); }
    const H = pipe.hidden, enc = await pipe.encode(m.prompt), un = m.guidance !== 1 ? (await pipe.encode(m.negative || '')).emb : null; let cond = enc.emb;
    if (m.steer && m.steer.alpha) { const pool = async (t) => { const e = await pipe.encode(t), v = new Float32Array(H); for (let i = 0; i < e.n; i++) for (let h = 0; h < H; h++) v[h] += e.emb[i * H + h] / e.n; return v; };
      const a = await pool(m.steer.plus), b = await pool(m.steer.minus); cond = steer(cond, a.map((v, i) => v - b[i]), m.steer.alpha, [...Array(77).keys()], H); }
    const swap = m.swap?.prompt ? { k: m.swap.k, emb: (await pipe.encode(m.swap.prompt)).emb } : null; const T0 = performance.now();
    if (m.op === 'attribute') { // token ablation at step 1: per-token latent-MSE + a spatial |Δx̂0| map per token
      const pad = cond.subarray(76 * H, 77 * H), step0 = async (c) => { let r; await pipe.generate(c, { steps: m.steps, cfg: m.guidance, uncond: un, seed: m.seed, onStep: (i, t, x0) => { r = x0; throw 'stop'; } }).catch((z) => { if (z !== 'stop') throw z; }); return r; };
      const base = await step0(cond), rows = [];
      for (let i = 1; i < enc.n - 1; i++) { const fill = new Float32Array(77 * H); for (let j = 0; j < 77 * H; j++) fill[j] = pad[j % H]; const x = await step0(ablateToken(cond, i, H, fill)), map = new Float32Array(64 * 64); let s = 0;
        for (let c = 0; c < 4; c++) for (let p = 0; p < 4096; p++) { const d = (x[c * 4096 + p] - base[c * 4096 + p]) ** 2; map[p] += d; s += d; }
        rows.push({ token: enc.tokens[i], mse: s / 16384, map }); post({ stage: 'attr', i, of: enc.n - 2 }); }
      return post({ done: true, tokens: rows, base: await img(base), ms: performance.now() - T0 }); }
    const lat = await pipe.generate(cond, { steps: m.steps, cfg: m.guidance, uncond: un, seed: m.seed, swap, onStep: m.preview ? async (i, t, x0) => post({ step: i, t, preview: await img(x0) }) : null });
    const out = await img(lat, m.decoder || 'vae'); post({ done: true, image: out, ms: performance.now() - T0, tokens: enc.tokens, timing: Object.fromEntries(Object.entries(pipe.timing).map(([k, v]) => [k, v.at(-1)])) });
  } catch (e) { post({ error: String(e?.message || e) }); } };
