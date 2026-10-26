# PROGRESS (times PT) — for resumption
- 06:37 start. Studied /workspace/neuralese (style, docs, tsotchke/qrng WASM, CLAIMS pattern). gh CLI not authed → GitHub connector (push_files, $file:) + Actions.
- 06:45 torch CPU in .venv; JDK 21; Android SDK at ~/android-sdk (platform 35, build-tools 35).
- 06:50 train/{data,model,train}.py: one 630k-param conditional UNet (self+cross attn at bottleneck) per modality; 3000 steps, batch 128, 2 threads each.
- 06:55 site/engine.js pure-JS forward: parity 1.4e-6 vs PyTorch. docs/PREREG.md frozen (sha256 51a18ca9…).
- 07:10 app.js (12 tabs), linalg.js, modalities.js, PWA, Capacitor android+ios, workflows android.yml / ios.yml; repo created.
- 07:15 SAE retrained (train/sae.py, λ=1.0) for image/video/shape3d; evals running (tools/eval.mjs, N=40).
- 07:30 shape3d+image evals done; README, RELEASE_NOTES, CLAIMS (generated), pages.yml; restore-binaries step in all workflows; gh-pages live placeholder. Pushing code to main.
