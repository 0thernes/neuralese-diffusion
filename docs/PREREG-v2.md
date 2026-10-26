# PREREG v2 — hypotheses frozen BEFORE running the real pretrained models and BEFORE evaluating the v2 self-trained models

Frozen 2026-10-26 ~09:00 PT. The SHA-256 of this file is recorded in PROGRESS.md and checked by `test/v2.test.mjs`; results live in
`site/results2/*.json` and `docs/CLAIMS.md` (v2 section). A hypothesis that fails stays in CLAIMS as ○. Thresholds are not tuned afterwards.

## Tier A — real pretrained diffusion models (ONNX, run by onnxruntime)
Prompts: the fixed list P8 = ["a red apple on a wooden table", "a lighthouse on a cliff at sunset", "a cat wearing sunglasses",
"a bowl of ramen, studio photo", "a snowy mountain village", "an astronaut riding a horse", "a vase of yellow tulips", "a city street at night in the rain"].
Seeds 0..7 (one per prompt). Resolution 512×512 latents 4×64×64 unless noted. Image metrics are computed on 8-bit RGB.

- R1 TAESD fidelity: for tiny-sd (20 steps), TAESD decode of the final latent vs the full SD VAE decode has median PSNR ≥ 18 dB over P8.
- R2 x̂0 lens: for tiny-sd 20 steps, the TAESD-decoded x̂0 at step 10 has median PSNR ≥ 15 dB vs the TAESD-decoded final x̂0.
- R3 token attribution: replacing the subject-noun token embedding (apple, lighthouse, cat, ramen, village, astronaut, tulips, street) by the
  padding-token embedding changes the step-1 predicted x̂0 (latent MSE) at least 2× more than replacing the token "a"/"an" (median over P8, tiny-sd).
- R4 prompt patching (prompt-to-prompt-style conditioning swap): run prompt A for the first k of 20 steps, then prompt B. With k=4, the final image is
  closer (pixel MSE) to pure-B than to pure-A in ≥ 6/8 pairs; with k=16 it is closer to pure-A in ≥ 6/8 pairs. Pairs: P8[i] → P8[(i+1)%8].
- R5 embedding steering: adding α·(E("red") − E("blue")) (mean-pooled token-embedding difference, broadcast over tokens) to the prompt embedding of
  "a car parked on a street" for α ∈ {−1, 0, 1, 2} raises the image's mean (R − B) channel monotonically (Spearman ρ = 1) for ≥ 3 of 4 seeds (sd-turbo 1-step).
- R6 CFG sweep: for tiny-sd, mean pixel std of the final image increases from guidance 1 to guidance 7.5 in ≥ 6/8 prompts.
- R7 latency (box CPU, 8 vCPU, onnxruntime-node, fp32 copies of the fp16 graphs): sd-turbo 1 step 512² end-to-end < 60 s; tiny-sd 20 steps 512² < 300 s.
  Browser WebGPU latency is reported only where actually measured; none is claimed otherwise.
- R8 Stable Audio 3 small (q4 ONNX): a 5 s clip at 8 steps runs end-to-end on box CPU in < 10 min, output RMS > 0.01, and two different prompts give
  clips whose log-mel spectrogram L2 distance is > 2× the distance between two seeds of the same prompt. (If the pipeline cannot be made to run, R8 is ○ with the reason.)

## Tier B — v2 self-trained models (~2.5M params, real datasets, 5400 s CPU budget each)
- S1 judge accuracy of DDIM-20, g=3 samples ≥ 0.70 for image (Fashion-MNIST), audio (FSDD), video (Moving-MNIST digit), 3D (ModelNet10).
- S2 FID-lite (Fréchet distance in judge-feature space, n=500) of g=3 samples < that of g=0 samples, per modality.
- S3 JS engine parity: max |ε_JS − ε_PyTorch| ≤ 1e-4 on the stored reference input (fp16 weights, f32 compute).
- S4 the v1 H1–H7 battery (tools/eval.mjs, unchanged thresholds, N=40) is re-run on the v2 models; prediction: H1, H5, H6 hold in all 4; H7 still fails in ≥ 3/4.
- S5 audio round-trip: Griffin-Lim (32 iters) inversion of real held-out spectrograms, re-analysed, keeps judge accuracy ≥ 0.8.
