// Real Stable-Diffusion-family pipeline over onnxruntime (node or web). Pass the `ort` module in.
// Pieces: CLIP text encoder -> UNet (eps-pred) with DDIM (deterministic) -> TAESD or full VAE decode.
// Interpretability hooks: per-step x0 preview, token ablation, prompt swap at step k, embedding steering, CFG sweeps.
const F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer);
export function f2h(v) { F32[0] = v; const x = U32[0], s = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff;
  if (e <= 0) { if (e < -10) return s; m |= 0x800000; const sh = 14 - e; let r = m >> sh; if ((m >> (sh - 1)) & 1) r++; return s | r; }
  if (e >= 31) return s | 0x7c00; let r = s | (e << 10) | (m >> 13); if (m & 0x1000) r++; return r; }
export function h2f(h) { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; if (e === 0) return s * m * 2 ** -24; if (e === 31) return m ? NaN : s * Infinity; return s * (1 + m / 1024) * 2 ** (e - 15); }
export const toF32 = (t) => (t.type === 'float16' ? Float32Array.from(t.data, h2f) : Float32Array.from(t.data));
export function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
export function randn(n, seed) { const r = mulberry(seed), o = new Float32Array(n); for (let i = 0; i < n; i += 2) { const u = Math.max(r(), 1e-12), v = r(), R = Math.sqrt(-2 * Math.log(u)); o[i] = R * Math.cos(2 * Math.PI * v); if (i + 1 < n) o[i + 1] = R * Math.sin(2 * Math.PI * v); } return o; }
export function alphasCumprod(T = 1000, b0 = 0.00085, b1 = 0.012) { const a = new Float64Array(T); let p = 1; for (let i = 0; i < T; i++) { const b = (Math.sqrt(b0) + (Math.sqrt(b1) - Math.sqrt(b0)) * i / (T - 1)) ** 2; p *= 1 - b; a[i] = p; } return a; }
export function timesteps(steps, spacing = 'leading', T = 1000) { if (spacing === 'trailing') return Array.from({ length: steps }, (_, i) => Math.round(T - i * T / steps) - 1);
  const r = Math.floor(T / steps); return Array.from({ length: steps }, (_, i) => (steps - 1 - i) * r + 1); }

export class SDPipeline {
  // sessions: {text, unet, vae?, taesd?}; dtype per session detected from input metadata names
  constructor(ort, sessions, tok, { latentScale = 0.18215, spacing = 'leading', hidden = 768 } = {}) { Object.assign(this, { ort, s: sessions, tok, latentScale, spacing, hidden }); this.ac = alphasCumprod(); this.timing = {}; }
  inType(sess, name) { const m = sess.inputMetadata; if (Array.isArray(m)) { const x = m.find((q) => q.name === name); if (x) return x.type; } return sess.__types?.[name] || 'float32'; }
  tensor(sess, name, f32, dims) { const t = this.inType(sess, name); if (t === 'int64') return new this.ort.Tensor('int64', BigInt64Array.from(f32, (v) => BigInt(Math.round(v))), dims);
    if (t === 'float16') { const u = new Uint16Array(f32.length); for (let i = 0; i < f32.length; i++) u[i] = f2h(f32[i]); return new this.ort.Tensor('float16', u, dims); } return new this.ort.Tensor('float32', Float32Array.from(f32), dims); }
  async tick(k, fn) { const t = performance.now(); const r = await fn(); (this.timing[k] ||= []).push(performance.now() - t); return r; }
  async encode(prompt) { const { ids, n, tokens } = this.tok.encode(prompt); const out = await this.tick('text', () => this.s.text.run({ input_ids: new this.ort.Tensor('int32', Int32Array.from(ids), [1, 77]) }));
    return { emb: toF32(out.last_hidden_state), n, tokens }; }
  async eps(x, t, emb) { const u = this.s.unet; const o = await this.tick('unet', () => u.run({ sample: this.tensor(u, 'sample', x, [1, 4, 64, 64]), timestep: this.tensor(u, 'timestep', [t], [1]), encoder_hidden_states: this.tensor(u, 'encoder_hidden_states', emb, [1, 77, this.hidden]) })); return toF32(o.out_sample); }
  // TAESD consumes the UNet-space (scaled) latents directly; the full VAE wants latents / 0.18215
  async decode(lat, which = 'taesd') { const sess = this.s[which]; const z = which === 'taesd' ? lat : lat.map((v) => v / this.latentScale); const o = await this.tick(which, () => sess.run({ latent_sample: this.tensor(sess, 'latent_sample', z, [1, 4, 64, 64]) }));
    const im = toF32(o.sample); return im; } // both this TAESD export and the VAE output [-1,1] (checked empirically: range and PSNR vs VAE)
  // opts: steps, cfg, seed, uncond emb, onStep(i,t,x0), swap {k, emb}, steer {vec, alpha, tokens:[idx]}
  async generate(cond, opts = {}) { const { steps = 20, cfg = 7.5, seed = 0, onStep, swap, uncond } = opts; let x = randn(4 * 64 * 64, seed); const ts = timesteps(steps, this.spacing);
    for (let i = 0; i < ts.length; i++) { const t = ts[i], ab = this.ac[t], abp = i + 1 < ts.length ? this.ac[ts[i + 1]] : 1; const c = swap && i >= swap.k ? swap.emb : cond;
      let e = await this.eps(x, t, c); if (cfg !== 1 && uncond) { const eu = await this.eps(x, t, uncond); for (let j = 0; j < e.length; j++) e[j] = eu[j] + cfg * (e[j] - eu[j]); }
      const x0 = new Float32Array(x.length); for (let j = 0; j < x.length; j++) x0[j] = (x[j] - Math.sqrt(1 - ab) * e[j]) / Math.sqrt(ab);
      if (onStep) await onStep(i, t, x0); for (let j = 0; j < x.length; j++) x[j] = Math.sqrt(abp) * x0[j] + Math.sqrt(1 - abp) * e[j]; }
    return x; }
}
// embedding ops (77 x H row-major)
export function ablateToken(emb, idx, H, fill) { const o = Float32Array.from(emb); for (let h = 0; h < H; h++) o[idx * H + h] = fill ? fill[idx * H + h] : 0; return o; }
export function steer(emb, vec, alpha, positions, H) { const o = Float32Array.from(emb); for (const p of positions) for (let h = 0; h < H; h++) o[p * H + h] += alpha * vec[h]; return o; }
export function toRGBA(img, W = 512) { const n = W * W, o = new Uint8ClampedArray(n * 4); for (let i = 0; i < n; i++) { for (let c = 0; c < 3; c++) o[i * 4 + c] = Math.round(Math.min(1, Math.max(0, (img[c * n + i] + 1) / 2)) * 255); o[i * 4 + 3] = 255; } return o; }
