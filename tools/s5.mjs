// S5 (PREREG-v2): Griffin-Lim (32 it) inversion of real held-out spectrograms, re-analysed with the JS STFT, judge accuracy ≥ 0.8.
import fs from 'node:fs'; import { loadModel, T3, argmax } from '../site/engine.js'; import { specToMag, griffinLim, stft, magToSpec } from '../site/modalities2.js';
const M = await loadModel('audio', new URL('../site/models2/', import.meta.url)), { logmin, logmax } = M.meta.spec, H = JSON.parse(fs.readFileSync('site/results2/heldout_audio.json'));
let direct = 0, rt = 0, sc = 0; for (let i = 0; i < H.X.length; i++) { const x = Float32Array.from(H.X[i]); direct += argmax(M.classify(T3(1, 64, 32, x))) === H.Y[i];
  const w = griffinLim(specToMag(x, logmin, logmax), 32, i + 1), S = stft(w), y = magToSpec(S.re, S.im, logmin, logmax); rt += argmax(M.classify(T3(1, 64, 32, y))) === H.Y[i];
  let num = 0, den = 0; for (let j = 0; j < x.length; j++) { const a = Math.exp((x[j] + 1) / 2 * (logmax - logmin) + logmin), b = Math.exp((y[j] + 1) / 2 * (logmax - logmin) + logmin); num += (a - b) ** 2; den += a * a; } sc += Math.sqrt(num / den); }
const n = H.X.length, R = { S5: { n, judge_acc_direct: direct / n, judge_acc_after_griffin_lim: rt / n, mean_spectral_convergence: sc / n, pass: rt / n >= 0.8 }, model_sha256: M.meta.sha256 }; console.log(JSON.stringify(R)); fs.writeFileSync('site/results2/s5_audio.json', JSON.stringify(R, null, 1));
