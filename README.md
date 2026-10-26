# Neuralese Diffusion

Sibling of [Neuralese](https://0thernes.github.io/neuralese/): an interpretability lab for **tiny diffusion models** that run entirely on your phone or browser.
Four modalities, one architecture (class-conditional ε-prediction UNet, ~630k params, 2.7 MB fp32, T=200, CFG):

| modality | data | classes |
|---|---|---|
| image | MNIST 16×16 | 10 digits |
| music | piano-roll 1×16×32 | arp-up, arp-down, chord, bass-alt |
| video | 4×16×16 moving blob | right, left, down, up |
| 3D | 64-point cloud (3×8×8) | sphere, cube, torus, cone |

Trained by us in PyTorch (`train/`), exported to JSON+base64, and run by a dependency-free JS forward pass (`site/engine.js`, parity vs PyTorch ≤2e-6).

**Live:** https://0thernes.github.io/neuralese-diffusion/ · **Apps:** see Releases (Android APK, iOS simulator .app, unsigned .ipa).

## Lab tabs
SAMPLE (DDPM/DDIM, steps, guidance, QRNG seed) · LENS (per-step x̂0) · TRAJ3D (PCA trajectories, three.js) · ACT3D (activations/weights) · ATTN (self & cross-attention) · SURGERY (low-rank/zero edits with undo) · STEER (mean-difference vectors) · PATCH (activation patching between seeds) · SCHEDULE (β/ᾱ/SNR plots) · SAE·PROBE · METRICS · DOCS.

## Honesty
Preregistered hypotheses (`docs/PREREG.md`, hash-frozen, checked by tests) → measured results in `docs/CLAIMS.md`, including negatives. Anything not working is in `docs/ROADMAP.md`.

## Run
```
node tools/restore-binaries.mjs   # binaries are stored as base64 in tools/binaries.json
npm ci && npm test
python3 -m http.server -d site 8000
```
Retrain: `python train/train.py <mod> && python train/sae.py <mod>`; evaluate: `node tools/eval.mjs <mod> && node tools/summarize.mjs && node tools/claims.mjs`.
CI: `.github/workflows/{pages,android,ios}.yml`. iOS signing optional via secrets `IOS_P12_BASE64`, `IOS_P12_PASSWORD`, `IOS_PROVISION_PROFILE_BASE64`, `IOS_TEAM_ID`.

Docs: BIBLE · GLOSSARY · COURSE · CLAIMS · STACK · PREREG · ROADMAP · PROGRESS. QRNG: Tsotchke quantum_rng (see LICENSE in site/tsotchke).
