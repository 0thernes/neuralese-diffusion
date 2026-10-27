# Neuralese Diffusion v3.0.0 (prerelease)

**Live:** https://0thernes.github.io/neuralese-diffusion/ · **Real tier:** https://0thernes.github.io/neuralese-diffusion/real/

## What's new in 3.0.0
- **Linux desktop app (x64):** an AppImage and a .deb built with Electron (`desktop/`). It bundles the full web app and all self-trained models, so it works offline. The `/real/` pretrained tier downloads its ONNX models at runtime and checks them against the same pinned SHA-256s. CI smoke-tests the packaged app under Xvfb.
- **LAB node importance, ACDC-lite and loss slice run in a Web Worker,** which fixes the v2 page freeze. They use a copy of the current weights, so surgery edits still count.
- Version is 3.0.0 everywhere (web, Android versionCode 3, iOS MARKETING_VERSION 3.0.0, desktop).

## Assets
- `NeuraleseDiffusion-3.0.0-x86_64.AppImage`: run `chmod +x` and then run the file. Some distros need `libfuse2`.
- `NeuraleseDiffusion-3.0.0-amd64.deb`: install with `sudo apt install ./NeuraleseDiffusion-3.0.0-amd64.deb`.
- `neuralese-diffusion-debug.apk`: Android debug build.
- `NeuraleseDiffusion-unsigned.ipa`: unsigned. Re-sign it with AltStore or Sideloadly before installing.
- `NeuraleseDiffusion-simulator.app.zip`: for the iOS Simulator (Xcode).

## Honest limitations
- The desktop builds are unsigned and x64 only. They have no auto-update.
- The self-trained models are still small and CPU-trained. Video generation is at chance, and R4 failed. See docs/CLAIMS-v2.md.
- The REAL tier needs a decent GPU (WebGPU) and downloads hundreds of MB to more than 1 GB on first use.
- The iOS IPA is unsigned (no Apple signing secrets), and the Android APK is a debug build.
