# Neuralese Diffusion v2.0.0 (prerelease)

**Live:** https://0thernes.github.io/neuralese-diffusion/ · **Real tier:** https://0thernes.github.io/neuralese-diffusion/real/

## What's new
- **Real pretrained tier (in-browser, onnxruntime-web, WebGPU with WASM fallback):** tiny-sd and sd-turbo text-to-image with a TAESD x̂0 preview, attribution, patching, CFG sweep and steering, plus Stable Audio (measured offline; see docs/MODELS.md). Every file is sha256-pinned, and size and latency are measured. Phones get the small tier.
- **Self-trained v2 tier (~2.5M-param UNets)** on real datasets for image, audio, video and 3D point clouds, with JS↔PyTorch parity tests, held-out FID-lite and judge accuracy.
- **v2 LAB tab:** node importance, ACDC-lite, loss slice, low-rank attention, quantization, ROME edit + test, CKA, sonification.
- **Docs:** CLAIMS-v2, MODELS, ROADMAP (100 items with falsifiable tests), plus STACK/GLOSSARY/COURSE/BIBLE updates. CLI: `node tools/nd.mjs`.

## Honest limitations
- The self-trained models are small and CPU-trained under a 1-hour budget. Image FID-lite with guidance is worse than without it, and video generation is at chance (judge accuracy 0.10). See docs/CLAIMS-v2.md.
- Hypothesis R4 FAILED (see CLAIMS-v2).
- The iOS IPA is unsigned unless signing secrets are present. The Android APK is a debug build.

Assets built by CI: Android debug APK, iOS simulator .app zip and an unsigned IPA.
