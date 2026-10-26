# PROGRESS (times PT) — for resumption
- 06:37 start. Studied /workspace/neuralese (style, docs, tsotchke/qrng WASM, CLAIMS pattern). gh CLI not authed → GitHub connector (push_files, $file:) + Actions.
- 06:45 torch CPU in .venv; JDK 21; Android SDK at ~/android-sdk (platform 35, build-tools 35).
- 06:50 train/{data,model,train}.py: one 630k-param conditional UNet (self+cross attn at bottleneck) per modality; 3000 steps, batch 128, 2 threads each.
- 06:55 site/engine.js pure-JS forward: parity 1.4e-6 vs PyTorch. docs/PREREG.md frozen (sha256 51a18ca9…).
- 07:10 app.js (12 tabs), linalg.js, modalities.js, PWA, Capacitor android+ios, workflows android.yml / ios.yml; repo created.
- 07:15 SAE retrained (train/sae.py, λ=1.0) for image/video/shape3d; evals running (tools/eval.mjs, N=40).
- 07:30 shape3d+image evals done; README, RELEASE_NOTES, CLAIMS (generated), pages.yml; restore-binaries step in all workflows; gh-pages live placeholder. Pushing code to main.
- 07:36 Source on main via Actions-unpacked bundle (connector caps $file args at 4/call); workflows pushed (abc29ee) → pages/android/ios running.
- 07:42 CI fix: node 22 needs test glob; concurrency groups. image/video evals in; CLAIMS regenerated.
- 07:50 Per parent: mobile workflows dispatch/tag-only, pages path-filtered, per-ref concurrency, Node24 actions; iOS+pages green on 7957e39; android setup-android removed; dispatched android+ios once.
- 07:58 All CI green; release v1.0.0 has APK, sim .app.zip, unsigned .ipa. Live smoke all 4 modalities OK (24 shots). Music eval still running.
- 08:13 Music eval done; summary + CLAIMS regenerated; fixed ATTN/SAE grid label for non-square bottleneck (music 4×8).

## v2.0.0 (started 08:38 PT)
- 08:45 Research: real ONNX diffusion candidates measured from HF file trees: sd-turbo (schmuell/sd-turbo-ort-web, fp16, unet 1733MB + text 681MB + vae-dec 99MB), tiny-sd fp16 (RanaLLC, unet 647 + text 246 + vae 99MB), TAESD (julienkay/taesd, 4.9MB dec), Stable Audio 3 small music q4 (lsb/..., ~640MB). No ONNX Shap-E/Point-E/AnimateDiff found.
- 08:51 v2 self-trained tier: train2/{model2,data2,train2}.py — 2.5M-param UNet, real datasets (Fashion-MNIST, FSDD spoken digits STFT, Moving-MNIST procedure on MNIST, ModelNet10 from ModelNet40-2048). Training 4×5400 s budget started (logs /workspace/data/train2_*.log). Outputs site/models2/.
- 08:52 docs/PREREG-v2.md frozen, sha256 cab4ccd125b0547948c74ce0304e6dd305918431c2bb18192bd7749d3e2b378c (R1–R8 real models, S1–S5 v2 self-trained).
- 09:22 PT: real tier working on box. `site/real/` = clip-tokenizer.js (BPE, parity with HF on 10 prompts), sd.js (DDIM pipeline + hooks),
  models.js (all files pinned: HF commit + sha256 + bytes; ORT web 1.23.0 bundle+glue+wasm sha-pinned), worker.js, index.html/real.js (UI).
  tiny-sd fp16 runs on onnxruntime-node CPU directly (no fp32 copies needed); sd-turbo needs graphOptimizationLevel=basic.
  Browser self-test (runtime + TAESD) passes locally on WASM and SwiftShader-WebGPU, max |Δ| vs box 2e-6. Hypothesis runner
  /workspace/ortbench/rtiny.mjs running (R1-R4,R6) → site/results2/real_tinysd.json; rturbo.mjs (R5) queued (OOM if concurrent).
  engine.js generalised (fp16 weights, E/M/heads/ctx from meta, nearest upsample to skip size, generic node hooks, ctx override).
  site/lab2.js: node ablation, ACDC-lite, low-rank SVD, int-k quantization, ROME-style ca.v edit, linear CKA, loss slice, sonify. test/v2.test.mjs 12/12.
- 09:31 PT: site/modalities2.js (torch-exact STFT, max err 2.3e-6 vs torch; ISTFT; Griffin-Lim; 3×16×16→points). app.js: v2 modalities image2/audio2/video2/pc3d2 in selector, LAB tab (node importance, ACDC-lite, low-rank, quantize, ROME, CKA, loss slice, sonify), v2 metrics table, link to real/. tools/export2.py → site/results2/heldout_*.json (+audio spec into models2/audio.json after training). /workspace/ortbench/rsa3.mjs (Stable Audio 3 R8 runner, bos=2 prepended) written; runs after rtiny. Training still running (ends ≈10:21 PT).
- 09:36 PT: tools/nd.mjs CLI (info/sample/parity/lab) works; docs/ROADMAP.md generated (100 items: 73 done, 3 partial, 20 planned, 4 won't) by tools/roadmap.py; shader-f16 badge in real UI.
