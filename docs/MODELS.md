# MODELS

Every remote file is pinned by Hugging Face commit + sha256 + byte size (site/real/models.js) and verified in the browser before use.

| model | params | download | tier | licence | measured latency (box CPU) |
|---|---|---|---|---|---|
| tiny-sd (Segmind, distilled SD-1.5) (RanaLLC/tiny-sd-onnx-fp16@2ceee652) | ≈0.55B total (UNet 323M fp16) | 993 MB | desktop (WebGPU or WASM) | CreativeML OpenRAIL-M | 20 steps + CFG: median 125 s |
| SD-Turbo (Stability AI, adversarial-distilled SD-2.1, 1–4 steps) (schmuell/sd-turbo-ort-web@ace89b7d) | ≈1.3B total (UNet 866M fp16) | 2514 MB | desktop WebGPU only | Stability AI Non-Commercial Research Community License | 1 step: median 8.8 s |
| TAESD decoder (julienkay/taesd) | ≈1.2 M | 4.9 MB | all incl. phones | MIT | browser WASM decode ≈ 4.8 s (64² latent), box ref |Δ| 1.4e-6 |
| Stable Audio 3 small music q4 (lsb/stable-audio-3-small-music-onnx) | ≈0.5 B | ≈650 MB | box only (needs a Gemma tokenizer in JS) | Stability Community + Gemma terms | 5 s clip, 8 steps: 14.2 s |
| onnxruntime-web 1.23.0 (runtime) | — | 25.5 MB | all | MIT | — |
| image2 self-trained (Fashion-MNIST) | 2.50 M | 5.4 MB fp16 | all incl. phones | MIT (weights), dataset licences in STACK.md | 1793 steps in 3602.2 s |
| audio2 self-trained (FSDD spoken digits) | 2.50 M | 5.4 MB fp16 | all incl. phones | MIT (weights), dataset licences in STACK.md | 762 steps in 3604.3 s |
| video2 self-trained (Moving-MNIST procedure on MNIST) | 2.50 M | 5.4 MB fp16 | all incl. phones | MIT (weights), dataset licences in STACK.md | 1325 steps in 3600.2 s |
| pc3d2 self-trained (ModelNet10 (from ModelNet40-2048)) | 2.50 M | 5.4 MB fp16 | all incl. phones | MIT (weights), dataset licences in STACK.md | 8076 steps in 5400.2 s |

Not shipped: no ONNX text-to-video or text-to-3D model small enough for a browser was found on Hugging Face (2026-10-26); video and 3D stay on the self-trained tier.
