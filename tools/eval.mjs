// Runs the pre-registered hypotheses (docs/PREREG.md) on one modality with the shipped JS engine. Usage: node tools/eval.mjs image [N]
import { loadModel, T3, argmax, steerHook, patchRun } from '../site/engine.js';
import { motionDir, gridToPoints, shapeJudge } from '../site/modalities.js';
import fs from 'node:fs'; import crypto from 'node:crypto';
const mod = process.argv[2], N = +(process.argv[3] || 40), v2 = mod.endsWith('2'), m = v2 ? await loadModel(mod.slice(0, -1), new URL('../site/models2/', import.meta.url)) : await loadModel(mod), K = m.K, { C, H, W } = m.meta, T = m.T;
const judge = (d) => argmax(m.classify(T3(C, H, W, d))), t0 = Date.now();
const run = (o) => m.sample({ steps: 20, scheduler: 'ddim', guidance: 3, ...o });
const acc = (f) => { let c = 0; for (let i = 0; i < N; i++) c += f(i) ? 1 : 0; return c / N; };
const R = { modality: mod, N, prereg_sha256: crypto.createHash('sha256').update(fs.readFileSync('docs/PREREG.md')).digest('hex') };
const base = []; for (let i = 0; i < N; i++) base.push(run({ seed: 1000 + i, cls: i % K }));
R.acc_g3 = acc((i) => judge(base[i].x.d) === i % K);
R.acc_g0 = acc((i) => judge(run({ seed: 1000 + i, cls: i % K, guidance: 0 }).x.d) === i % K);
R.H1 = R.acc_g3 >= 0.8 && R.acc_g3 - R.acc_g0 >= 0.4;
R.lens_agree = acc((i) => { const tr = base[i].traj, s = tr.find((q) => q.t <= 0.6 * T); return judge(s.x0) === judge(base[i].x.d); }); R.H2 = R.lens_agree >= 0.7;
const pt = (early) => acc((i) => { const a = i % K, b = (a + 1) % K; return judge(patchRun(m, { steps: 20, scheduler: 'ddim', guidance: 3, seed: 1000 + i, cls: a }, { steps: 20, scheduler: 'ddim', guidance: 3, seed: 1000 + i, cls: b }, (t) => early ? t >= T / 2 : t < T / 2).x.d) === a; });
R.flip_early = pt(true); R.flip_late = pt(false); R.H3 = R.flip_early - R.flip_late >= 0.3;
R.steer_hit = acc((i) => judge(run({ seed: 1000 + i, cls: K, midHook: steerHook(m, i % K, 4) }).x.d) === i % K);
R.uncond_hit = acc((i) => judge(run({ seed: 1000 + i, cls: K }).x.d) === i % K); R.H4 = R.steer_hit >= 0.5;
R.probe_acc = m.meta.probe_heldout_acc_tHalf; R.H5 = R.probe_acc >= 2 / K;
R.acc_ddpm50 = acc((i) => judge(m.sample({ steps: 50, scheduler: 'ddpm', guidance: 3, seed: 1000 + i, cls: i % K }).x.d) === i % K); R.H6 = Math.abs(R.acc_g3 - R.acc_ddpm50) <= 0.1;
m.edit('net.ca.o.weight', (d) => d.fill(0), 'zero ca.o.weight'); m.edit('net.ca.o.bias', (d) => d.fill(0), 'zero ca.o.bias');
R.acc_noCA = acc((i) => judge(run({ seed: 1000 + i, cls: i % K }).x.d) === i % K); m.undoLast(); m.undoLast(); R.H7 = R.acc_g3 - R.acc_noCA >= 0.3;
if (mod === 'video') { R.motion_acc = acc((i) => motionDir(base[i].x.d) === i % K); R.H8 = R.motion_acc >= 0.8; }
if (mod === 'shape3d') { R.chamfer_acc = acc((i) => shapeJudge(gridToPoints(base[i].x.d)) === i % K); R.H9 = R.chamfer_acc >= 0.6; }
R.ms_per_forward = (Date.now() - t0) / (N * 20 * 2 * (1 + 1 + 3 + 3 + 0.5 + 0.5 + 2.5 + 1)); R.seconds = (Date.now() - t0) / 1000;
const od = v2 ? 'site/results2' : 'site/results', of = v2 ? `eval_${mod}.json` : `${mod}.json`; if (v2) R.model_sha256 = m.meta.sha256; fs.mkdirSync(od, { recursive: true }); fs.writeFileSync(`${od}/${of}`, JSON.stringify(R, null, 1)); console.log(JSON.stringify(R));
