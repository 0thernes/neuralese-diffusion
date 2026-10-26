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
- 10:24 PT: shipped web increment via stage-v2 branch (bundle + unpack workflow → single commit on main, then pages dispatch). models2 not included yet (training finishing).
- 10:27 PT: LIVE: increment on main + Pages deployed (unpack-stage, pages green). Live real/ self-test WASM: TAESD load 4.9 s, decode 4.8 s, max|Δ| 1.37e-6 (shots/v2/live-real-*).
- 10:40 PT: rtiny done: R1 median PSNR(TAESD,VAE) 25.1 dB pass; R2 median 22.4 pass; R3 median ratio 15.9 pass. pc3d training DONE (gen judge acc g3 0.994, FID-lite 80 vs floor 41). rturbo.mjs (R5) started nohup → ortbench/rturbo.log.
- 10:41 PT: R4 FAIL (k=4 closer to B only 1/8; k=16 closer to A 8/8 — layout locks in within 4 of 20 steps); R6 PASS 8/8.
- 10:55 PT: v2 run1 DONE image (2755 steps, gen acc 0.29, FID-lite 591 — undertrained) / video (2078 steps, gen acc 0.09 — fail) / pc3d (8076 steps, gen acc 0.994). R5 PASS (4/4 seeds monotone). train2.py now checkpoints every 500 it + resumes. Retraining image+video 3600 s × 3 threads (nohup, logs /workspace/data/train2b_*.log); run1 outputs backed up in /workspace/data/v2run1.
- 11:02 PT: audio run1 DONE (980 steps only, gen acc 0.49) → retraining audio 3600 s × 2 thr with checkpoints. Tests: v2.test 17/17 (S3 parity all 4 models). S5 PASS: Griffin-Lim round-trip judge acc 0.9375 = direct 0.9375 (n=32) → site/results2/s5_audio.json (re-run after retrain).
- 11:05 PT: tools/eval.mjs supports v2 (mod2 → results2/eval_<mod>2.json); S4 eval pc3d2 N=40 started (nohup).
- 11:00 PT (box clock; earlier 10:55/11:02/11:05 entries were mis-stamped, real ≈10:3x): R5 PASS 4/4; R7 PASS (sd-turbo 1-step unet+VAE ≈ 8.5–9.6 s, tiny-sd 20 steps CFG ≈ 105–130 s, box CPU under training load); R8 PASS: Stable Audio 3 small q4 5 s clip 8 steps ≈ 13–14 s total, RMS ≥ 0.061, prompt/seed log-mel distance ratio 6.3. WAVs → site/results2/audio/.

## 2026-10-26 11:14 PT — increment 2 shipped
- Bundle (11×3MB parts) pushed sequentially to stage-v2; trigger → unpack-stage success → main ad6c51f "v2 results: …" (single commit) → pages success.
- Retraining (train2b, nohup, ckpt every 500 it) still running: image it800/1203s, video it600/1298s, audio it200/848s (CPU contended). eval pc3d2 N=40 running.
- Next: on retrain finish → export2, s5, eval ×4, claims2, npm test, ship final commit; live smoke of v2 mods + LAB; tag v2.0.0.

## 2026-10-26 12:13 PT — retrain (run 2) finished
- CPU-contended, fewer steps than hoped. image 1793 steps g3 acc 0.37 (run1 0.286), FID-lite g3 585 vs g0 537 (floor 3.1) — still worse than g0;
  audio 762 steps acc 0.48 (run1 0.486); video 1325 steps acc 0.10 (chance; run1 0.09). Honest: only image improved; video gen fails.
- 12:16 PT export2 + S5 (0.8125→0.8125 PASS) + npm test 43/43; evals image2/audio2/video2 N=16 started (nohup); pc3d2 N=40 still running.
- 12:16 PT release workflows retargeted to v2.0.0 (local), RELEASE_NOTES v2 written.
- 12:50 PT live smoke (inc2, run1 weights) in progress: image2 sample/LENS/LAB/METRICS OK; header v1→v2 fix; smoke2 now also runs LAB node-importance.
- app versions bumped to 2.0.0 (package, Android, iOS).
- 12:53 PT evals (eval.mjs) not finished after ~2h on contended CPU → killed; using train2b DONE metrics (python, n=500) for CLAIMS.
- 12:58 PT CLAIMS-v2/MODELS regenerated with plain verdicts (video at chance, image FID worse w/ CFG, S4 not completed); npm test 43/43.
- 13:00 PT pushed ci commit ab2b83e (release workflows → v2.0.0) via connector (GITHUB_TOKEN can't edit workflows).
