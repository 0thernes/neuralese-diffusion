// Pure-JS forward pass of the tiny conditional UNet (train/model.py) + DDPM/DDIM samplers with hooks.
// No WebGPU, no ONNX: every weight is a Float32Array you can read and edit. Batch size 1; CFG runs the net twice.
const isNode = typeof process !== 'undefined' && !!process.versions?.node;
export async function loadModel(name, base = new URL('./models/', import.meta.url)) {
  const rd = async (f) => isNode ? (await import('node:fs')).readFileSync(new URL(f, base), 'utf8') : await (await fetch(new URL(f, base))).text();
  const meta = JSON.parse(await rd(name + '.json')); let b64 = ''; for (let i = 0; i < (meta.b64parts || 1); i++) b64 += (await rd(meta.b64parts ? `${name}.${i}.b64` : name + '.b64')).trim();
  const bin = typeof Buffer !== 'undefined' && isNode ? Buffer.from(b64, 'base64') : Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const u8 = new Uint8Array(bin.length); u8.set(bin); const all = meta.dtype === 'f16' ? f16ToF32(new Uint16Array(u8.buffer)) : new Float32Array(u8.buffer);
  const got = await sha256(u8); if (got !== meta.sha256) throw new Error(`${name} weights sha256 ${got} ≠ ${meta.sha256}`);
  const W = {}; for (const m of meta.manifest) W[m.name] = { d: all.subarray(m.offset, m.offset + m.size), shape: m.shape };
  return new Model(meta, W, all);
}
export function h2f(h) { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); }
export function f16ToF32(u16) { const o = new Float32Array(u16.length); for (let i = 0; i < u16.length; i++) o[i] = h2f(u16[i]); return o; }
export async function sha256(u8) { const h = await crypto.subtle.digest('SHA-256', u8); return [...new Uint8Array(h)].map((x) => x.toString(16).padStart(2, '0')).join(''); }
export const T3 = (c, h, w, d) => ({ c, h, w, d: d || new Float32Array(c * h * w) });
export function conv(x, Wt, bt, stride = 1) {
  const [co, ci, k] = Wt.shape, p = (k - 1) >> 1, W = Wt.d, b = bt.d, H = x.h, Wd = x.w, oh = Math.floor((H + 2 * p - k) / stride) + 1, ow = Math.floor((Wd + 2 * p - k) / stride) + 1;
  const y = T3(co, oh, ow), X = x.d, Y = y.d;
  for (let o = 0; o < co; o++) {
    const yo = o * oh * ow; Y.fill(b[o], yo, yo + oh * ow);
    for (let i = 0; i < ci; i++) {
      const xi = i * H * Wd;
      for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const wv = W[((o * ci + i) * k + ky) * k + kx]; if (wv === 0) continue;
        for (let yy = 0; yy < oh; yy++) {
          const sy = yy * stride + ky - p; if (sy < 0 || sy >= H) continue; const row = xi + sy * Wd, orow = yo + yy * ow;
          for (let xx = 0; xx < ow; xx++) { const sx = xx * stride + kx - p; if (sx >= 0 && sx < Wd) Y[orow + xx] += wv * X[row + sx]; }
        }
      }
    }
  }
  return y;
}
export function gnorm(x, g, b, G = 8, eps = 1e-5) {
  const n = x.h * x.w, cpg = x.c / G, y = T3(x.c, x.h, x.w);
  for (let q = 0; q < G; q++) {
    let s = 0, s2 = 0; const a = q * cpg * n, e = a + cpg * n;
    for (let i = a; i < e; i++) { s += x.d[i]; } const m = s / (cpg * n); for (let i = a; i < e; i++) { const v = x.d[i] - m; s2 += v * v; }
    const inv = 1 / Math.sqrt(s2 / (cpg * n) + eps);
    for (let c = q * cpg; c < (q + 1) * cpg; c++) for (let i = c * n; i < (c + 1) * n; i++) y.d[i] = (x.d[i] - m) * inv * g.d[c] + b.d[c];
  }
  return y;
}
export const silu = (x) => { const y = T3(x.c, x.h, x.w); for (let i = 0; i < x.d.length; i++) { const v = x.d[i]; y.d[i] = v / (1 + Math.exp(-v)); } return y; };
const relu = (x) => { const y = T3(x.c, x.h, x.w); for (let i = 0; i < x.d.length; i++) y.d[i] = x.d[i] > 0 ? x.d[i] : 0; return y; };
export function linear(v, Wt, bt) { const [o, i] = Wt.shape, y = new Float32Array(o); for (let r = 0; r < o; r++) { let s = bt ? bt.d[r] : 0; for (let c = 0; c < i; c++) s += Wt.d[r * i + c] * v[c]; y[r] = s; } return y; }
const add = (a, b) => { const y = T3(a.c, a.h, a.w); for (let i = 0; i < a.d.length; i++) y.d[i] = a.d[i] + b.d[i]; return y; };
// nearest upsample to an explicit size (torch F.interpolate(size=, mode='nearest'): src = floor(dst * in / out)); v1 used exact 2x
const upTo = (x, H, W) => { const y = T3(x.c, H, W); for (let c = 0; c < x.c; c++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) y.d[(c * H + i) * W + j] = x.d[(c * x.h + Math.floor(i * x.h / H)) * x.w + Math.floor(j * x.w / W)]; return y; };
const cat = (a, b) => { const y = T3(a.c + b.c, a.h, a.w); y.d.set(a.d); y.d.set(b.d, a.d.length); return y; };
export function tembed(t, d = 32) { const h = d / 2, o = new Float32Array(d); for (let i = 0; i < h; i++) { const f = Math.exp(-Math.log(10000) * i / h), a = t * f; o[i] = Math.fround(Math.sin(a)); o[i + h] = Math.fround(Math.cos(a)); } return o; }
export function rng(seed) { let s = seed >>> 0; const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; let spare = null; return { u, n() { if (spare !== null) { const v = spare; spare = null; return v; } const a = Math.max(u(), 1e-12), b = u(), r = Math.sqrt(-2 * Math.log(a)); spare = r * Math.sin(2 * Math.PI * b); return r * Math.cos(2 * Math.PI * b); } }; }
export class Model {
  constructor(meta, W, all) {
    this.meta = meta; this.E = meta.E || 128; this.Mc = meta.M || 64; this.heads = meta.heads || 2; this.nctx = meta.ctx || 4; this.W = W; this.all = all; this.K = meta.ncls; this.T = meta.T; this.undo = []; this.pristine = all.slice();
    const [b0, b1] = meta.beta; this.beta = new Float64Array(this.T); this.ab = new Float64Array(this.T); let p = 1;
    for (let t = 0; t < this.T; t++) { this.beta[t] = Math.fround(b0 + (b1 - b0) * t / (this.T - 1)); p *= 1 - this.beta[t]; this.ab[t] = Math.fround(p); }
  }
  w(n) { const t = this.W['net.' + n] || this.W[n]; if (!t) throw new Error('no tensor ' + n); return t; }
  res(p, x, e, rec, opt) {
    let h = conv(silu(gnorm(x, this.w(p + '.n1.weight'), this.w(p + '.n1.bias'))), this.w(p + '.c1.weight'), this.w(p + '.c1.bias'));
    const te = linear(e, this.w(p + '.t.weight'), this.w(p + '.t.bias')), n = h.h * h.w; for (let c = 0; c < h.c; c++) for (let i = 0; i < n; i++) h.d[c * n + i] += te[c];
    h = conv(silu(gnorm(h, this.w(p + '.n2.weight'), this.w(p + '.n2.bias'))), this.w(p + '.c2.weight'), this.w(p + '.c2.bias'));
    const sk = this.W['net.' + p + '.sk.weight'] ? conv(x, this.w(p + '.sk.weight'), this.w(p + '.sk.bias')) : x; let y = add(h, sk); if (opt?.hooks?.[p]) y = opt.hooks[p](y) || y; if (rec) rec[p] = y; return y;
  }
  attn(p, x, ctx, rec, opt) {
    const C = x.c, N = x.h * x.w, H = this.heads, dh = C / H, hn = gnorm(x, this.w(p + '.n.weight'), this.w(p + '.n.bias'));
    const tok = []; for (let i = 0; i < N; i++) { const v = new Float32Array(C); for (let c = 0; c < C; c++) v[c] = hn.d[c * N + i]; tok.push(v); }
    const src = ctx || tok, q = tok.map((v) => linear(v, this.w(p + '.q.weight'), this.w(p + '.q.bias'))), k = src.map((v) => linear(v, this.w(p + '.k.weight'), this.w(p + '.k.bias'))), vv = src.map((v) => linear(v, this.w(p + '.v.weight'), this.w(p + '.v.bias')));
    const M = src.length, maps = [], out = tok.map(() => new Float32Array(C)), sc = 1 / Math.sqrt(dh);
    for (let hh = 0; hh < H; hh++) {
      const A = new Float32Array(N * M);
      for (let i = 0; i < N; i++) { let mx = -Infinity; for (let j = 0; j < M; j++) { let s = 0; for (let d = 0; d < dh; d++) s += q[i][hh * dh + d] * k[j][hh * dh + d]; A[i * M + j] = s * sc; if (A[i * M + j] > mx) mx = A[i * M + j]; } let z = 0; for (let j = 0; j < M; j++) { A[i * M + j] = Math.exp(A[i * M + j] - mx); z += A[i * M + j]; } for (let j = 0; j < M; j++) A[i * M + j] /= z; }
      if (opt?.ablateHead?.[p] === hh) A.fill(0);
      maps.push(A);
      for (let i = 0; i < N; i++) for (let j = 0; j < M; j++) { const a = A[i * M + j]; if (a) for (let d = 0; d < dh; d++) out[i][hh * dh + d] += a * vv[j][hh * dh + d]; }
    }
    if (rec) rec[p + '.maps'] = { maps, N, M, heads: H };
    const y = T3(C, x.h, x.w); y.d.set(x.d);
    for (let i = 0; i < N; i++) { const o = linear(out[i], this.w(p + '.o.weight'), this.w(p + '.o.bias')); for (let c = 0; c < C; c++) y.d[c * N + i] += o[c]; }
    return y;
  }
  // forward(x: T3, t: int, y: class id or K for null) -> eps (T3). opt.rec collects activations; opt.midHook(h) may edit the bottleneck.
  forward(x, t, y, opt = {}) {
    const rec = opt.rec || null, w = (n) => this.w(n);
    const e1 = linear(tembed(t), w('te1.weight'), w('te1.bias')).map((v) => v / (1 + Math.exp(-v)));
    const e = linear(e1, w('te2.weight'), w('te2.bias')), ce = w('cls.weight'); const E = this.E; for (let i = 0; i < E; i++) { const v = e[i] + ce.d[y * E + i]; e[i] = v / (1 + Math.exp(-v)); }
    const cw = w('ctx.weight'), ctx = []; const Mc = this.Mc, nc = this.nctx; for (let k = 0; k < nc; k++) ctx.push(cw.d.slice(y * nc * Mc + k * Mc, y * nc * Mc + (k + 1) * Mc));
    const h0 = conv(x, w('cin.weight'), w('cin.bias')); if (rec) rec.cin = h0;
    const hk = (n, v) => (opt.hooks?.[n] ? opt.hooks[n](v) || v : v);
    const h1 = this.res('d1', h0, e, rec, opt); let h = conv(h1, w('ds1.weight'), w('ds1.bias'), 2); if (rec) rec.ds1 = h;
    const h2 = this.res('d2', h, e, rec, opt); h = conv(h2, w('ds2.weight'), w('ds2.bias'), 2); if (rec) rec.ds2 = h;
    h = this.res('m1', h, e, rec, opt); h = hk('sa', this.attn('sa', h, null, rec, opt)); if (rec) rec.sa = h; h = hk('ca', this.attn('ca', h, opt.ctx || ctx, rec, opt)); if (rec) rec.ca = h; h = this.res('m2', h, e, rec, opt);
    if (opt.midHook) h = opt.midHook(h) || h; if (rec) rec.mid = h;
    h = conv(upTo(h, h2.h, h2.w), w('us2.weight'), w('us2.bias')); h = this.res('u2', cat(h, h2), e, rec, opt);
    h = conv(upTo(h, h1.h, h1.w), w('us1.weight'), w('us1.bias')); h = this.res('u1', cat(h, h1), e, rec, opt);
    return conv(silu(gnorm(h, w('no.weight'), w('no.bias'))), w('cout.weight'), w('cout.bias'));
  }
  classify(x) { const w = (n) => this.W['clf.' + n]; let h = relu(conv(x, w('c1.weight'), w('c1.bias'))); h = relu(conv(h, w('c2.weight'), w('c2.bias'), 2)); h = relu(conv(h, w('c3.weight'), w('c3.bias'), 2)); const v = new Float32Array(h.c), n = h.h * h.w; for (let c = 0; c < h.c; c++) { let s = 0; for (let i = 0; i < n; i++) s += h.d[c * n + i]; v[c] = s / n; } return linear(v, w('fc.weight'), w('fc.bias')); }
  probe(mid) { const v = poolMid(mid), mu = this.W['probe.mu'].d, sd = this.W['probe.sd'].d; for (let i = 0; i < v.length; i++) v[i] = (v[i] - mu[i]) / sd[i]; return linear(v, this.W['probe.weight'], this.W['probe.bias']); }
  sae(vec) { return linear(vec, this.W['sae.enc.weight'], this.W['sae.enc.bias']).map((v) => v > 0 ? v : 0); }
  steerVec(target) { const m = this.W['steer.means'].d, a = this.W['steer.mall'].d, D = a.length, v = new Float32Array(D); for (let i = 0; i < D; i++) v[i] = m[target * D + i] - a[i]; return v; }
  timesteps(steps) { const ts = []; for (let i = 0; i < steps; i++) ts.push(Math.round((this.T - 1) * (1 - i / steps))); return [...new Set(ts)]; }
  // sample({seed, steps, scheduler:'ddpm'|'ddim', guidance, cls, record, midHook(h,t), onStep}) -> {x, traj:[{t,xt,x0}], acts}
  sample(o = {}) { const it = this.sampleIter(o); let r; while (!(r = it.next()).done); return r.value; }
  async sampleAsync(o = {}) { const it = this.sampleIter(o); let r; while (!(r = it.next()).done) await new Promise((f) => setTimeout(f, 0)); return r.value; }
  *sampleIter(o = {}) {
    const { C, H, W } = this.meta, K = this.K, r = rng(o.seed ?? 1), steps = o.steps ?? 50, g = o.guidance ?? 3, cls = o.cls ?? 0, sch = o.scheduler || 'ddim';
    const ts = this.timesteps(sch === 'ddpm' && !o.steps ? this.T : steps);
    let x = T3(C, H, W); for (let i = 0; i < x.d.length; i++) x.d[i] = o.init ? o.init[i] : r.n();
    const traj = [], acts = [];
    for (let si = 0; si < ts.length; si++) {
      const t = ts[si], tp = si + 1 < ts.length ? ts[si + 1] : -1, rec = o.record ? {} : null, hkc = o.midHook ? { midHook: (h) => o.midHook(h, t, si, 'c') } : {}, hku = o.midHook ? { midHook: (h) => o.midHook(h, t, si, 'u') } : {};
      const ec = this.forward(x, t, cls, { rec, ...hkc, ablateHead: o.ablateHead }), eps = T3(C, H, W);
      if (g !== 1 && cls !== K) { const en = this.forward(x, t, K, { ...hku, ablateHead: o.ablateHead }); for (let i = 0; i < eps.d.length; i++) eps.d[i] = en.d[i] + g * (ec.d[i] - en.d[i]); } else eps.d.set(ec.d);
      const a = this.ab[t], ap = tp >= 0 ? this.ab[tp] : 1, x0 = T3(C, H, W);
      for (let i = 0; i < x0.d.length; i++) x0.d[i] = Math.max(-1, Math.min(1, (x.d[i] - Math.sqrt(1 - a) * eps.d[i]) / Math.sqrt(a)));
      const nx = T3(C, H, W);
      if (sch === 'ddim') { for (let i = 0; i < nx.d.length; i++) { const e2 = (x.d[i] - Math.sqrt(a) * x0.d[i]) / Math.sqrt(1 - a); nx.d[i] = Math.sqrt(ap) * x0.d[i] + Math.sqrt(1 - ap) * e2; } }
      else { const bt = 1 - a / ap, var_ = tp >= 0 ? (1 - ap) / (1 - a) * bt : 0; for (let i = 0; i < nx.d.length; i++) { const mean = Math.sqrt(ap) * bt / (1 - a) * x0.d[i] + Math.sqrt(1 - bt) * (1 - ap) / (1 - a) * x.d[i]; nx.d[i] = mean + (tp >= 0 ? Math.sqrt(var_) * r.n() : 0); } }
      traj.push({ t, xt: x.d.slice(), x0: x0.d.slice() }); if (rec) acts.push({ t, rec }); o.onStep?.(si, ts.length, x0); yield si;
      x = tp >= 0 ? nx : x0;
    }
    return { x, traj, acts, ts };
  }
  // ---- weight surgery with exact undo ----
  edit(name, fn, label) { const t = this.W[name]; const before = t.d.slice(); fn(t.d, t.shape); this.undo.push({ name, before, label }); return label; }
  undoLast() { const u = this.undo.pop(); if (u) this.W[u.name].d.set(u.before); return u?.label; }
  reset() { this.all.set(this.pristine); this.undo = []; }
}
export function poolMid(mid) { const n = mid.h * mid.w, v = new Float32Array(mid.c); for (let c = 0; c < mid.c; c++) { let s = 0; for (let i = 0; i < n; i++) s += mid.d[c * n + i]; v[c] = s / n; } return v; }
export const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
export function softmax(a) { const m = Math.max(...a), e = [...a].map((v) => Math.exp(v - m)), z = e.reduce((s, v) => s + v, 0); return e.map((v) => v / z); }
// Steering hook: add alpha * (mu_target - mu_all) to every bottleneck position.
export const steerHook = (model, target, alpha, pred = () => true) => { const v = model.steerVec(target); return (h, t) => { if (!pred(t)) return h; const n = h.h * h.w; for (let c = 0; c < h.c; c++) for (let i = 0; i < n; i++) h.d[c * n + i] += alpha * v[c]; return h; }; };
// Activation patching: run source, cache bottleneck per step; in target run, replace bottleneck at steps where pred(t) is true.
export function patchRun(model, src, tgt, pred) {
  const cache = {}; model.sample({ ...src, midHook: (h, t, si, br) => { cache[si + br] = h.d.slice(); return h; } });
  return model.sample({ ...tgt, midHook: (h, t, si, br) => { if (pred(t) && cache[si + br]) h.d.set(cache[si + br]); return h; } });
}
