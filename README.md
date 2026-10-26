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

## v2.0.0 — real models + a v12-style lab

- **Real pretrained tier** (`/real/`): tiny-sd (0.55 B) and SD-Turbo (1.3 B) run **in your browser** via onnxruntime-web (WebGPU, WASM fallback). Every file is pinned by Hugging Face commit + sha256 and verified before use. A 4.9 MB TAESD decoder gives a live x̂₀ preview at every step. Tools: token attribution by ablation, prompt patching, CFG sweep, embedding steering. Phones get the small tier (TAESD self-test + self-trained models); the big models need a desktop GPU.
- **Self-trained v2 tier**: 2.5 M-param UNets on real datasets — Fashion-MNIST, FSDD spoken digits (spectrogram + Griffin-Lim playback), Moving-MNIST video, ModelNet10 point clouds. The JS engine matches PyTorch to ≤ 1e-4.
- **LAB tab**: node ablation, ACDC-lite circuits, SVD low-rank, int-k quantization, ROME-style cross-attention edit, linear CKA, loss-landscape slice, sonification.
- **CLI**: `node tools/nd.mjs info | sample image2 --cls 3 | parity pc3d2 | lab image2 acdc`.
- Results, failures included: [docs/CLAIMS-v2.md](docs/CLAIMS-v2.md) · models and licences: [docs/MODELS.md](docs/MODELS.md) · 100-item roadmap: [docs/ROADMAP.md](docs/ROADMAP.md) · pre-registration: [docs/PREREG-v2.md](docs/PREREG-v2.md).
- Stable Audio 3 small (q4) runs on the box (5 s clip in about 14 s on CPU) but not yet in the browser, because it needs a JS Gemma tokenizer. No browser-sized ONNX video or 3D generator exists, so those stay self-trained.

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
