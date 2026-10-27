// v2 modality decoders (DOM-free). Audio: log-magnitude STFT <-> waveform via Griffin-Lim (phase is NOT generated — it is
// re-estimated; the model produces only the magnitude spectrogram). Point clouds: 3×16×16 grid -> 256 points.
export const SPEC = { sr: 8000, nfft: 126, hop: 125, n: 4000, bins: 64, frames: 32 };
const hann = (N) => Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)); // periodic, as torch.hann_window(periodic=True)
// torch.stft(center=True, pad_mode='reflect') magnitude, first `frames` frames; returns [bins][frames] (re, im too)
export function stft(x, { nfft = SPEC.nfft, hop = SPEC.hop, frames = SPEC.frames } = {}) { const w = hann(nfft), p = nfft >> 1, N = x.length, xp = new Float64Array(N + 2 * p);
  for (let i = 0; i < xp.length; i++) { let j = i - p; if (j < 0) j = -j; if (j >= N) j = 2 * (N - 1) - j; xp[i] = x[j]; }
  const B = (nfft >> 1) + 1, re = [], im = []; for (let k = 0; k < B; k++) { re.push(new Float64Array(frames)); im.push(new Float64Array(frames)); }
  for (let f = 0; f < frames; f++) for (let k = 0; k < B; k++) { let a = 0, b = 0; for (let n = 0; n < nfft; n++) { const v = xp[f * hop + n] * w[n], ph = -2 * Math.PI * k * n / nfft; a += v * Math.cos(ph); b += v * Math.sin(ph); } re[k][f] = a; im[k][f] = b; } return { re, im }; }
export function istft(re, im, { nfft = SPEC.nfft, hop = SPEC.hop, n = SPEC.n } = {}) { const w = hann(nfft), p = nfft >> 1, F = re[0].length, B = re.length, L = (F - 1) * hop + nfft, y = new Float64Array(L), ws = new Float64Array(L);
  for (let f = 0; f < F; f++) for (let t = 0; t < nfft; t++) { let s = 0; for (let k = 0; k < B; k++) { const c = k === 0 || (nfft % 2 === 0 && k === nfft / 2) ? 1 : 2, ph = 2 * Math.PI * k * t / nfft; s += c * (re[k][f] * Math.cos(ph) - im[k][f] * Math.sin(ph)); } y[f * hop + t] += s / nfft * w[t]; ws[f * hop + t] += w[t] * w[t]; }
  const o = new Float32Array(n); for (let i = 0; i < n; i++) { const j = i + p; o[i] = j < L && ws[j] > 1e-8 ? y[j] / ws[j] : 0; } return o; }
// model output d (64×32 in [-1,1]) -> magnitude using the training normalisation
export function specToMag(d, logmin, logmax) { const B = SPEC.nfft / 2 + 1, M = []; for (let k = 0; k < B; k++) { M.push(new Float64Array(SPEC.frames)); if (k < SPEC.bins) for (let f = 0; f < SPEC.frames; f++) M[k][f] = Math.max(0, Math.exp((d[k * SPEC.frames + f] + 1) / 2 * (logmax - logmin) + logmin) - 1e-2); } return M; }
export function magToSpec(re, im, logmin, logmax) { const o = new Float32Array(SPEC.bins * SPEC.frames); for (let k = 0; k < SPEC.bins; k++) for (let f = 0; f < SPEC.frames; f++) o[k * SPEC.frames + f] = (Math.log(Math.hypot(re[k][f], im[k][f]) + 1e-2) - logmin) / (logmax - logmin) * 2 - 1; return o; }
export function griffinLim(M, iters = 32, seed = 1) { let s = seed >>> 0; const r = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); const B = M.length, F = M[0].length;
  let re = M.map((row) => row.map(() => 0)), im = M.map((row) => row.map(() => 0)); for (let k = 0; k < B; k++) for (let f = 0; f < F; f++) { const ph = 2 * Math.PI * r(); re[k][f] = M[k][f] * Math.cos(ph); im[k][f] = M[k][f] * Math.sin(ph); }
  let x = istft(re, im); for (let it = 0; it < iters; it++) { const S = stft(x); for (let k = 0; k < B; k++) for (let f = 0; f < F; f++) { const a = Math.hypot(S.re[k][f], S.im[k][f]) || 1e-12; re[k][f] = M[k][f] * S.re[k][f] / a; im[k][f] = M[k][f] * S.im[k][f] / a; } x = istft(re, im); } return x; }
export const gridToPoints2 = (d, n = 256) => { const P = []; for (let i = 0; i < n; i++) P.push([d[i], d[n + i], d[2 * n + i]]); return P; };
