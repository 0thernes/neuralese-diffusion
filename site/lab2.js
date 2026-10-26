// v2 interpretability toolkit for the white-box JS UNet (engine.js). DOM-free: used by the UI, tools/eval2.mjs and tests.
// Diffusion equivalents of the Neuralese v12 LM tools. Every metric is a held-out denoising loss or a judge readout — no fake numbers.
import { T3, rng } from './engine.js';
export const NODES = ['d1', 'd2', 'm1', 'sa', 'ca', 'm2', 'u2', 'u1'];
// held-out denoising loss E||eps - eps_hat(x_t,t,y)||^2 at fixed (seeded) t and noise; opt passes hooks/ctx through
export function denoiseLoss(model, X, Y, { seed = 3, ts = [20, 60, 100, 140, 180], opt = {} } = {}) { const { C, H, W } = model.meta, n = C * H * W, r = rng(seed); let s = 0, k = 0;
  for (let i = 0; i < X.length; i++) for (const t of ts) { const a = model.ab[t], eps = new Float32Array(n), xt = T3(C, H, W); for (let j = 0; j < n; j++) { eps[j] = r.n(); xt.d[j] = Math.sqrt(a) * X[i][j] + Math.sqrt(1 - a) * eps[j]; }
    const e = model.forward(xt, t, Y[i], opt); for (let j = 0; j < n; j++) s += (e.d[j] - eps[j]) ** 2; k += n; } return s / k; }
// --- activation patching / ablation over the 8 computational nodes (zero or mean) ---
export function ablateHooks(nodes, mode = 'zero', means = {}) { const h = {}; for (const nd of nodes) h[nd] = (v) => { const y = T3(v.c, v.h, v.w); if (mode === 'mean' && means[nd]) { const m = means[nd], n = v.h * v.w; for (let c = 0; c < v.c; c++) y.d.fill(m[c], c * n, (c + 1) * n); } return y; }; return h; }
export function nodeMeans(model, X, Y, t = 100) { const acc = {}, { C, H, W } = model.meta; for (let i = 0; i < X.length; i++) { const rec = {}, x = T3(C, H, W, Float32Array.from(X[i])); model.forward(x, t, Y[i], { rec });
  for (const nd of NODES) { const v = rec[nd], n = v.h * v.w; acc[nd] ||= new Float32Array(v.c); for (let c = 0; c < v.c; c++) { let s = 0; for (let j = 0; j < n; j++) s += v.d[c * n + j]; acc[nd][c] += s / n / X.length; } } } return acc; }
export function nodeImportance(model, X, Y, mode = 'mean') { const means = mode === 'mean' ? nodeMeans(model, X, Y) : {}, base = denoiseLoss(model, X, Y); const out = { base, nodes: {} };
  for (const nd of NODES) out.nodes[nd] = denoiseLoss(model, X, Y, { opt: { hooks: ablateHooks([nd], mode, means) } }) - base; return out; }
// ACDC-lite: greedily remove nodes (mean-ablate) whose removal raises loss by < tau; the survivors are the "circuit" (node level, not edge level)
export function acdcLite(model, X, Y, tau = 0.01) { const means = nodeMeans(model, X, Y), base = denoiseLoss(model, X, Y), removed = []; let cur = base; const log = [];
  for (const nd of [...NODES].reverse()) { const l = denoiseLoss(model, X, Y, { opt: { hooks: ablateHooks([...removed, nd], 'mean', means) } }); log.push({ node: nd, loss: l, delta: l - cur }); if (l - cur < tau) { removed.push(nd); cur = l; } }
  return { base, final: cur, removed, kept: NODES.filter((n) => !removed.includes(n)), log, tau }; }
// --- weight surgery (all through model.edit -> exact undo) ---
export function jacobiSVD(A, m, n) { // one-sided Jacobi on A (m×n, m>=n not required); returns {U (m×n), S (n), V (n×n)}
  const U = Float64Array.from(A), V = new Float64Array(n * n); for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 30; sweep++) { let off = 0; for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) { let a = 0, b = 0, c = 0; for (let i = 0; i < m; i++) { const x = U[i * n + p], y = U[i * n + q]; a += x * x; b += y * y; c += x * y; }
    if (Math.abs(c) < 1e-12 * Math.sqrt(a * b) || c === 0) continue; off = Math.max(off, Math.abs(c) / Math.sqrt(a * b)); const z = (b - a) / (2 * c), t = Math.sign(z || 1) / (Math.abs(z) + Math.sqrt(1 + z * z)), cs = 1 / Math.sqrt(1 + t * t), sn = cs * t;
    for (let i = 0; i < m; i++) { const x = U[i * n + p], y = U[i * n + q]; U[i * n + p] = cs * x - sn * y; U[i * n + q] = sn * x + cs * y; } for (let i = 0; i < n; i++) { const x = V[i * n + p], y = V[i * n + q]; V[i * n + p] = cs * x - sn * y; V[i * n + q] = sn * x + cs * y; } }
    if (off < 1e-10) break; }
  const S = new Float64Array(n); for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += U[i * n + j] ** 2; S[j] = Math.sqrt(s); if (S[j] > 1e-300) for (let i = 0; i < m; i++) U[i * n + j] /= S[j]; } return { U, S, V }; }
// truncate a 2-D view (rows = out, cols = rest) of a weight to rank k
export function lowRank(model, name, k) { return model.edit(name, (d, shape) => { const m = shape[0], n = d.length / m, { U, S, V } = jacobiSVD(d, m, n), ord = [...S.keys()].sort((a, b) => S[b] - S[a]).slice(0, k);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) { let s = 0; for (const r of ord) s += U[i * n + r] * S[r] * V[j * n + r]; d[i * n + j] = s; } }, `rank-${k} ${name}`); }
export function quantize(model, names, bits) { for (const nm of names) model.edit(nm, (d) => { let mx = 0; for (const v of d) mx = Math.max(mx, Math.abs(v)); const q = 2 ** (bits - 1) - 1, s = mx / q || 1; for (let i = 0; i < d.length; i++) d[i] = Math.round(d[i] / s) * s; }, `int${bits} ${nm}`); return names.length; }
export const weightNames = (model, pred = () => true) => Object.keys(model.W).filter((k) => k.startsWith('net.') && model.W[k].shape.length >= 2 && pred(k));
// ROME-style rank-one edit of the cross-attention value projection: make class `src`'s 4 context tokens emit the values of class `tgt`'s tokens.
// Minimal-Frobenius-norm update W' = W + (V* − W K)(KᵀK)⁻¹Kᵀ with K = src tokens (Mc×4): exact on those keys, untouched on their orthogonal complement.
export function romeCA(model, src, tgt) { const Mc = model.Mc, nc = model.nctx, cw = model.w('ctx.weight').d, tok = (y, k) => cw.subarray(y * nc * Mc + k * Mc, y * nc * Mc + (k + 1) * Mc);
  const Wv = model.w('ca.v.weight'), Kt = [...Array(nc).keys()].map((k) => tok(src, k)), Vt = [...Array(nc).keys()].map((k) => tok(tgt, k));
  const G = new Float64Array(nc * nc); for (let a = 0; a < nc; a++) for (let b = 0; b < nc; b++) { let s = 0; for (let i = 0; i < Mc; i++) s += Kt[a][i] * Kt[b][i]; G[a * nc + b] = s; }
  const Gi = inv(G, nc), name = Object.keys(model.W).find((k) => k.endsWith('ca.v.weight'));
  return model.edit(name, (d) => { const R = []; for (let a = 0; a < nc; a++) { const r = new Float64Array(Mc); for (let o = 0; o < Mc; o++) { let wk = 0, wv = 0; for (let i = 0; i < Mc; i++) { wk += d[o * Mc + i] * Kt[a][i]; wv += d[o * Mc + i] * Vt[a][i]; } r[o] = wv - wk; } R.push(r); }
    for (let o = 0; o < Mc; o++) for (let i = 0; i < Mc; i++) { let s = 0; for (let a = 0; a < nc; a++) { let ki = 0; for (let b = 0; b < nc; b++) ki += Gi[a * nc + b] * Kt[b][i]; s += R[a][o] * ki; } d[o * Mc + i] += s; } }, `ROME ca.v: class ${src}→${tgt}`); }
function inv(A, n) { const M = Float64Array.from(A), I = new Float64Array(n * n); for (let i = 0; i < n; i++) { I[i * n + i] = 1; M[i * n + i] += 1e-6; }
  for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r; for (let j = 0; j < n; j++) { [M[c * n + j], M[p * n + j]] = [M[p * n + j], M[c * n + j]]; [I[c * n + j], I[p * n + j]] = [I[p * n + j], I[c * n + j]]; }
    const d = M[c * n + c]; for (let j = 0; j < n; j++) { M[c * n + j] /= d; I[c * n + j] /= d; } for (let r = 0; r < n; r++) if (r !== c) { const f = M[r * n + c]; for (let j = 0; j < n; j++) { M[r * n + j] -= f * M[c * n + j]; I[r * n + j] -= f * I[c * n + j]; } } } return I; }
// --- representation similarity: linear CKA between two activation sets (rows = samples) ---
export function linearCKA(X, Y) { const n = X.length, c = (Z) => { const d = Z[0].length, mu = new Float64Array(d); for (const z of Z) for (let j = 0; j < d; j++) mu[j] += z[j] / n; return Z.map((z) => Float64Array.from(z, (v, j) => v - mu[j])); };
  const A = c(X), B = c(Y), gram = (Z) => { const K = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { let s = 0; for (let k = 0; k < Z[i].length; k++) s += Z[i][k] * Z[j][k]; K[i * n + j] = K[j * n + i] = s; } return K; };
  const K = gram(A), L = gram(B); let kl = 0, kk = 0, ll = 0; for (let i = 0; i < n * n; i++) { kl += K[i] * L[i]; kk += K[i] * K[i]; ll += L[i] * L[i]; } return kl / Math.sqrt(kk * ll); }
export function pooledActs(model, X, Y, t, nodes = NODES) { const { C, H, W } = model.meta, out = Object.fromEntries(nodes.map((n) => [n, []])), r = rng(11);
  for (let i = 0; i < X.length; i++) { const a = model.ab[t], x = T3(C, H, W); for (let j = 0; j < x.d.length; j++) x.d[j] = Math.sqrt(a) * X[i][j] + Math.sqrt(1 - a) * r.n(); const rec = {}; model.forward(x, t, Y[i], { rec });
    for (const nd of nodes) { const v = rec[nd], n = v.h * v.w, p = new Float32Array(v.c); for (let c = 0; c < v.c; c++) { let s = 0; for (let j = 0; j < n; j++) s += v.d[c * n + j]; p[c] = s / n; } out[nd].push(p); } } return out; }
// --- loss landscape: 1-D slice along a filter-normalised random direction over all net weights ---
export function lossSlice(model, X, Y, alphas = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1], seed = 5) { const r = rng(seed), names = weightNames(model), dir = {};
  for (const nm of names) { const w = model.W[nm].d, d = Float32Array.from(w, () => r.n()); let nw = 0, nd = 0; for (let i = 0; i < w.length; i++) { nw += w[i] ** 2; nd += d[i] ** 2; } const s = Math.sqrt(nw / (nd || 1)); for (let i = 0; i < d.length; i++) d[i] *= s; dir[nm] = d; }
  const orig = Object.fromEntries(names.map((n) => [n, model.W[n].d.slice()])), out = [];
  for (const a of alphas) { for (const nm of names) { const w = model.W[nm].d, o = orig[nm], d = dir[nm]; for (let i = 0; i < w.length; i++) w[i] = o[i] + a * d[i]; } out.push({ alpha: a, loss: denoiseLoss(model, X, Y) }); }
  for (const nm of names) model.W[nm].d.set(orig[nm]); return out; }
// --- sonification: map an x̂0 trajectory to audio (pitch = mean |x̂0|, loudness = step change). Purely a listening aid. ---
export function sonify(traj, sr = 22050, perStep = 0.12) { const n = Math.floor(traj.length * perStep * sr), out = new Float32Array(n); let ph = 0;
  for (let s = 0; s < traj.length; s++) { const x = traj[s].x0, prev = s ? traj[s - 1].x0 : x; let m = 0, dv = 0; for (let i = 0; i < x.length; i++) { m += Math.abs(x[i]); dv += Math.abs(x[i] - prev[i]); } m /= x.length; dv /= x.length;
    const f = 220 * 2 ** (m * 3), amp = Math.min(0.3, 0.05 + dv); for (let i = 0; i < perStep * sr; i++) { ph += 2 * Math.PI * f / sr; out[Math.floor(s * perStep * sr) + i] = amp * Math.sin(ph) * Math.min(1, i / 300, (perStep * sr - i) / 300); } } return out; }
