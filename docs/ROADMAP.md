# ROADMAP (v3.0.0)

100 items. Status counts: done 72, partial 4, planned 20, won't 4. "done" means shipped and the test passes as of v2.0.0; see docs/CLAIMS.md for measured numbers.

| # | area | item | status | falsifiable test |
|---|---|---|---|---|
| 1 | real | tiny-sd fp16 ONNX runs in browser (WASM) | done | tools/smoke-real.mjs: self-test |Δ|<1e-4 vs box reference |
| 2 | real | tiny-sd on WebGPU | done | same self-test with #webgpu passes; latency logged in real UI |
| 3 | real | sd-turbo 1-step in browser (desktop WebGPU only) | partial | box run R5 passes; browser run needs ≥4 GB GPU — manual test on desktop |
| 4 | real | every remote file pinned by HF commit + sha256 + bytes | done | worker rejects any byte mismatch (unit: fetchVerified with wrong sha throws) |
| 5 | real | onnxruntime-web runtime sha-pinned and loaded from verified blobs | done | tampered glue → load fails |
| 6 | real | Cache API storage of verified models | done | second load makes 0 network requests for model files |
| 7 | real | download-size confirm dialog before any >10 MB fetch | done | UI test: no fetch before confirm |
| 8 | real | phone detection → small tier only, #force override | done | UA=iPhone → sd-turbo button disabled |
| 9 | real | JS CLIP BPE tokenizer | done | 10 prompts identical ids to HF CLIPTokenizer (V2 test) |
| 10 | real | DDIM with leading spacing for tiny-sd | done | timesteps(20) equals diffusers DDIMScheduler list |
| 11 | real | Euler trailing 1-step for sd-turbo | done | t=999 single step; output not noise (std>20 on 0-255) |
| 12 | real | TAESD x̂0 preview every step | done | R2: final TAESD vs VAE PSNR ≥ 20 dB |
| 13 | real | x̂0 lens strip in UI | done | strip has steps/2 frames |
| 14 | real | token attribution by ablation | done | R3: noun tokens Δ ≥ 2× article tokens (median) |
| 15 | real | prompt patching (swap embedding mid-run) | done | R4: early swap changes image more than late swap |
| 16 | real | CFG sweep | done | R6: pixel std increases from g=1 to g=7.5 |
| 17 | real | text-embedding steering | done | steer α=0 reproduces baseline bit-exactly |
| 18 | real | latency table measured, not quoted | done | site/results2/real_*.json has ms per op |
| 19 | real | Stable Audio 3 small q4 on box | partial | R8: two prompts produce log-mel distance > seed distance |
| 20 | real | Stable Audio 3 in browser | planned | needs JS Gemma SentencePiece tokenizer + 650 MB; test: browser output matches box within 1e-3 |
| 21 | real | pretrained video model | won't | no ONNX text-to-video ≤2 GB found (searched HF 2026-10-26); revisit if one appears |
| 22 | real | pretrained 3D model (Shap-E/Point-E) | won't | no ONNX export available; test would be: point cloud judge acc > chance |
| 23 | real | fp16 vs fp32 numeric comparison | planned | PSNR fp16 vs fp32 tiny-sd ≥ 35 dB |
| 24 | real | int8 quantized UNet tier for phones | planned | phone generate 256² in < 60 s |
| 25 | real | attention-map capture from real UNet | planned | requires graph surgery to expose attn probs; test: maps sum to 1 |
| 26 | real | LoRA loading | planned | LoRA scale 0 == base bit-exact |
| 27 | real | img2img | planned | strength 0 returns input |
| 28 | real | inpainting mask | planned | unmasked region PSNR > 30 dB vs input |
| 29 | real | negative prompt | done | uncond embedding = encoded negative prompt; empty == default |
| 30 | real | seed reproducibility across WASM/WebGPU | partial | same seed ⇒ PSNR ≥ 30 dB across EPs |
| 31 | v2 | 2.5M-param UNet on Fashion-MNIST | done | JS-vs-PyTorch parity < 1e-3 |
| 32 | v2 | spoken-digit spectrogram diffusion (FSDD) | done | parity < 1e-3 |
| 33 | v2 | Moving-MNIST 8-frame video diffusion | done | parity < 1e-3 |
| 34 | v2 | ModelNet10 point-cloud diffusion | done | parity < 1e-3 |
| 35 | v2 | fp16 weight storage | done | sha256 of fp16 blob matches meta |
| 36 | v2 | held-out ε-MSE reported | done | value present in models2/*.json |
| 37 | v2 | judge classifier accuracy on real test | done | S1: reported per modality |
| 38 | v2 | generated-sample judge accuracy | done | S2 vs chance 0.1 |
| 39 | v2 | FID-lite in judge features with real-vs-real floor | done | gen FID and floor both in meta |
| 40 | v2 | Griffin-Lim waveform reconstruction | done | S5: round-trip judge acc ≥ 0.8× direct |
| 41 | v2 | torch-exact STFT in JS | done | max err < 1e-4 vs torch.stft |
| 42 | v2 | longer training on GPU | planned | heldout loss decreases ≥10% with 10× steps |
| 43 | v2 | EMA weights | done | meta.ema=0.999 |
| 44 | v2 | v-prediction variant | planned | compare held-out loss at equal steps |
| 45 | v2 | DDPM/DDIM both available | done | both samplers produce finite output |
| 46 | v2 | latent diffusion on self-trained autoencoder | planned | reconstruction PSNR ≥ 25 dB |
| 47 | v2 | classifier-free guidance sweep | partial | S4 battery on v2 models not completed (stopped >2 h); g0 vs g3 measured in train2 |
| 48 | v2 | probe accuracy at t=T/2 | done | in meta |
| 49 | v2 | SAE on bottleneck | done | FVU, L0, dead features in meta |
| 50 | v2 | UI for all 4 v2 modalities | done | smoke screenshots per modality |
| 51 | lab | x̂0 lens | done | final x̂0 equals sample |
| 52 | lab | activation patching bottleneck | done | patch whole schedule ⇒ judge = source class majority |
| 53 | lab | node ablation importance | done | ablating all nodes raises loss > any single |
| 54 | lab | ACDC-lite node circuit | done | kept circuit loss within τ·|removed| of base |
| 55 | lab | edge-level ACDC | planned | edge circuit ⊆ node circuit |
| 56 | lab | attribution patching (gradient approx) | planned | correlates (ρ>0.5) with true ablation |
| 57 | lab | SAE features | done | FVU < 0.5 |
| 58 | lab | SAE feature steering | planned | clamping a feature changes judge P |
| 59 | lab | ROME-style cross-attn edit | done | after edit, src-class samples judged as tgt ≥ 50% |
| 60 | lab | MEMIT multi-edit | planned | two edits both hold |
| 61 | lab | SVD low-rank truncation | done | rank-full == original within 1e-4 |
| 62 | lab | int-k quantization sweep | done | int8 loss change < 5% |
| 63 | lab | linear probes | done | acc > chance |
| 64 | lab | linear CKA between nodes | done | CKA(X,X)=1 |
| 65 | lab | loss-landscape 1-D slice | done | minimum at α≈0 |
| 66 | lab | 2-D loss landscape | planned | grid convex near 0 |
| 67 | lab | steering vectors | done | α=0 identity |
| 68 | lab | sonification | done | output length = steps × 0.12 s |
| 69 | lab | weight surgery with exact undo | done | undo restores bit-exact |
| 70 | lab | 3D trajectory PCA | done | renders |
| 71 | lab | 3D activation volumes | done | renders |
| 72 | lab | attention maps | done | rows sum to 1 |
| 73 | lab | noise schedule plots | done | ᾱ monotone |
| 74 | lab | QRNG seeds with provenance | done | NIST subset reported |
| 75 | lab | causal scrubbing | planned | scrubbed loss within 10% of base for true hypothesis |
| 76 | lab | induction-like circuit search | won't | no sequence structure in these models |
| 77 | lab | logit lens analogue for real UNet | done | TAESD x̂0 lens |
| 78 | lab | memorisation check (nearest train neighbour) | planned | gen-to-train NN distance > test-to-train |
| 79 | lab | per-timestep importance curves | planned | importance varies across t |
| 80 | lab | weight histograms | planned | renders |
| 81 | tool | CLI tools/nd.mjs | done | `nd parity image` passes |
| 82 | tool | browser smoke script with screenshots | done | shots in neuralese-diffusion-shots/v2 |
| 83 | tool | unit tests v1 (26) + v2 | done | npm test green |
| 84 | tool | pre-registration v2 frozen by sha | done | V2-1 test checks sha |
| 85 | tool | auto-generated CLAIMS | done | claims.mjs output matches results |
| 86 | tool | PWA offline | done | reload offline serves app |
| 87 | tool | Android APK on tag | done | release asset present |
| 88 | tool | iOS simulator app + unsigned ipa on tag | done | release asset present |
| 89 | tool | signed iOS build | won't | needs Apple developer account credentials |
| 90 | tool | CI path filters and concurrency | done | docs-only commit triggers no build |
| 91 | docs | README v2 | done | links resolve |
| 92 | docs | BIBLE v2 chapters | done | covers every tab |
| 93 | docs | GLOSSARY v2 terms | done | every UI term defined |
| 94 | docs | COURSE v2 exercises | done | each exercise runnable in UI |
| 95 | docs | STACK | done | lists all deps with versions |
| 96 | docs | MODELS table with licences | done | each model has licence + sha |
| 97 | docs | CLAIMS v2 honest results incl. failures | done | every R/S hypothesis listed |
| 98 | docs | this ROADMAP | done | 100 items |
| 99 | docs | video tutorial | planned | — |
| 100 | real | WebGPU fp16 shader-f16 detection badge | done | badge shows shader-f16 yes/no from adapter.features |
