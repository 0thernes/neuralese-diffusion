# ROADMAP — not built (yet). Nothing here is claimed to work.
- Waveform / spectrogram audio diffusion (current music model is a symbolic piano roll + sine synth).
- Latent video diffusion with temporal attention; >4 frames; real video data (e.g. Moving-MNIST).
- Permutation-invariant point-cloud denoiser (PointNet/transformer); current 3D model is a CNN over a z/azimuth-ordered 8×8 grid.
- ONNX export + onnxruntime-web / WebGPU kernels for speed; current engine is scalar JS.
- Stable-Diffusion-class models (SD-Turbo, tiny-sd): too large for the "tiny, fully editable" goal; would only be a viewer.
- Text conditioning (CLIP); conditioning here is a class label + 4 learned context tokens.
- Gradient attribution (integrated gradients through the denoiser), ROME-style rank-one edits, training/fine-tuning on device.
- Real FID/KID; current quality metric is a tiny judge classifier + judge-free metrics for video/3D.
- Signed iOS distribution (TestFlight/App Store) — the CI only signs if you add certificate secrets.
