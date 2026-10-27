// v3: LAB analyses that run thousands of UNet forwards (node importance, ACDC-lite, loss slice) execute here so the page stays responsive.
// The worker receives the CURRENT weights (including any surgery edits) as a copy of model.all and rebuilds the model from meta.manifest.
import { Model } from './engine.js'; import * as LAB from './lab2.js';
self.onmessage = (e) => { const { id, fn, meta, all, X, Y, args } = e.data;
  try { const W = {}; for (const m of meta.manifest) W[m.name] = { d: all.subarray(m.offset, m.offset + m.size), shape: m.shape };
    const r = LAB[fn](new Model(meta, W, all), X, Y, ...(args || [])); self.postMessage({ id, ok: true, r: JSON.parse(JSON.stringify(r)) });
  } catch (err) { self.postMessage({ id, ok: false, e: String(err && err.stack || err) }); } };
