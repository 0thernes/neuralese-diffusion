#!/usr/bin/env node
// Neuralese-Diffusion CLI.  node tools/nd.mjs <cmd> [args]
//   sample <mod|mod2> [--cls k] [--steps n] [--g w] [--seed s] [--out f.json]   generate (v2 models: suffix 2, e.g. image2)
//   parity <mod|mod2>                                                          max |JS − PyTorch| on the stored reference
//   lab <mod2> importance|acdc|slice|quant:<bits>|rank:<k>                     LAB analyses on held-out data
//   info                                                                         list models with params/bytes/sha
import fs from 'node:fs'; import { loadModel, T3, argmax, softmax } from '../site/engine.js'; import * as LAB from '../site/lab2.js';
const [cmd, a1, ...rest] = process.argv.slice(2), opt = {}; for (let i = 0; i < rest.length; i++) if (rest[i].startsWith('--')) opt[rest[i].slice(2)] = rest[++i]; else opt._ = rest[i];
const root = new URL('../site/', import.meta.url), load = (m) => m.endsWith('2') ? loadModel(m.slice(0, -1), new URL('models2/', root)) : loadModel(m, new URL('models/', root));
const J = (o) => console.log(JSON.stringify(o, (k, v) => typeof v === 'number' ? +v.toPrecision(5) : v, 1));
if (cmd === 'info') { for (const [d, ms] of [['models', ['image', 'music', 'video', 'shape3d']], ['models2', ['image', 'audio', 'video', 'pc3d']]]) for (const m of ms) { try { const j = JSON.parse(fs.readFileSync(new URL(`${d}/${m}.json`, root))); J({ model: m + (d === 'models2' ? '2' : ''), params: j.params_unet, bytes: j.bytes, sha256: j.sha256, heldout_loss: j.heldout_loss }); } catch {} } }
else if (cmd === 'sample') { const M = await load(a1), o = { cls: +(opt.cls ?? 0), steps: +(opt.steps ?? 20), guidance: +(opt.g ?? 3), seed: +(opt.seed ?? 1), scheduler: 'ddim' }, r = M.sample(o), { C, H, W } = M.meta, p = softmax(M.classify(T3(C, H, W, r.x.d)));
  J({ ...o, label: M.meta.labels?.[o.cls], judge: M.meta.labels?.[argmax(p)] ?? argmax(p), p_judge: Math.max(...p) }); if (opt.out) fs.writeFileSync(opt.out, JSON.stringify({ shape: [C, H, W], x: [...r.x.d] })); }
else if (cmd === 'parity') { const M = await load(a1), r = M.meta.ref, e = M.forward(T3(M.meta.C, M.meta.H, M.meta.W, Float32Array.from(r.x)), r.t, r.y); let d = 0; e.d.forEach((v, i) => (d = Math.max(d, Math.abs(v - r.eps_cond[i])))); J({ model: a1, max_abs_diff: d, pass: d < 1e-3 }); }
else if (cmd === 'lab') { const M = await load(a1), H = JSON.parse(fs.readFileSync(new URL(`results2/heldout_${a1.replace(/2$/, '')}.json`, root))), X = H.X.slice(0, 8), Y = H.Y.slice(0, 8), [op, arg] = (opt._ || 'importance').split(':');
  if (op === 'importance') J(LAB.nodeImportance(M, X, Y)); else if (op === 'acdc') J(LAB.acdcLite(M, X, Y, +(arg || 0.01))); else if (op === 'slice') J(LAB.lossSlice(M, X, Y));
  else if (op === 'quant') { const b = LAB.denoiseLoss(M, X, Y); LAB.quantize(M, LAB.weightNames(M), +arg); J({ bits: +arg, base: b, after: LAB.denoiseLoss(M, X, Y) }); }
  else if (op === 'rank') { const b = LAB.denoiseLoss(M, X, Y), n = LAB.weightNames(M, (w) => /\.(sa|ca)\./.test(w)); n.forEach((w) => LAB.lowRank(M, w, +arg)); J({ k: +arg, tensors: n.length, base: b, after: LAB.denoiseLoss(M, X, Y) }); } }
else console.log(fs.readFileSync(new URL(import.meta.url)).toString().split('\n').slice(1, 7).join('\n'));
