# STACK — what is really used

| layer | what | status |
|---|---|---|
| training | PyTorch 2.x CPU (`train/*.py`) on the build box, 8 cores, no GPU | real |
| models | 4 × tiny class-conditional ε-prediction UNet, ≈630 k params, 2.7 MB float32 each (`site/models/*.b64` + `*.json` manifest, SHA-256 pinned) | real |
| inference | pure JavaScript (`site/engine.js`): conv, GroupNorm, SiLU, 2-head self-attention, cross-attention, DDPM/DDIM, CFG. No ONNX, no WebGPU | real, parity ≤ 2e-6 vs PyTorch |
| 3D | three.js r1xx vendored from Neuralese (`site/vendor/three`) | real |
| audio | WebAudio sine synth of the generated piano roll (`modalities.js`) | real; the model does **not** generate waveforms |
| randomness | Tsotchke `quantum_rng` (commit 1a77e77, MIT) compiled to WASM, reused verbatim from Neuralese; seeds the JS PRNG. Classical simulation, not hardware quantum randomness | real |
| PWA | `manifest.webmanifest` + cache-first `sw.js` | real (offline after first load) |
| mobile | Capacitor 7 (`android/`, `ios/`), webDir = `site/` | Android APK built on the box + CI; iOS built only in CI on `macos-latest` |
| CI | `.github/workflows/android.yml`, `ios.yml` → artifacts + GitHub Release v1.0.0 | real |

**Not used (roadmap only):** ONNX Runtime Web, WebGPU kernels, Stable Diffusion / SD-Turbo (≈1–2 GB fp16 even for the UNet; far
above "tiny"), Point-E, AudioLDM, video latent diffusion, Eshkol / Moonlab / QGTL (used in Neuralese, not here).

## Data and licences
| modality | data | licence |
|---|---|---|
| image | MNIST (LeCun, Cortes, Burges), 60 k train images, bilinear-downsampled to 16×16 | CC BY-SA 3.0 (per Yann LeCun's site / torchvision) |
| music | procedural (`train/data.py`, seed 0): 4 pattern classes on a 2-octave C-major piano roll | generated here, MIT |
| video | procedural 4-frame moving 4×4 blob, 4 directions | generated here, MIT |
| 3D | procedural 64-point clouds (sphere, cube surface, torus, cone), random z-rotation and scale | generated here, MIT |
