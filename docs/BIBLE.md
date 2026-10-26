# THE BIBLE — how Neuralese Diffusion works, end to end

## 1. Forward (noising) process
Pick T = 200 and a linear schedule β_t from 1e-4 to 0.02. With α_t = 1 − β_t and ᾱ_t = ∏_{s≤t} α_s,
any clean sample x₀ can be noised in one shot: **x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε**, ε ~ N(0, I).
At t = 199, ᾱ ≈ 0.13, so x_T still has a faint trace of x₀ (SCHEDULE tab shows this; a known flaw of short linear schedules).

## 2. The network
One architecture for all four modalities (only the input channel count and grid size change):
```
x (C×H×W) ─ conv3×3 → 32 ─ Res(32) ──────────────────────────────┐ skip h1
              └ conv s2 → 64 ─ Res(64) ──────────────────┐ skip h2 │
                    └ conv s2 → 64 ─ Res ─ SelfAttn(2 heads) ─ CrossAttn(2 heads, 4 ctx tokens) ─ Res = "mid"
                    ┌ up×2, conv ─ cat h2 ─ Res(128→64)
              ┌ up×2, conv ─ cat h1 ─ Res(96→32)
ε̂ ← conv3×3 ← SiLU ← GroupNorm
```
Time t enters every ResBlock through a sinusoidal embedding → MLP, plus a learned class embedding. The class (or ∅) also
selects 4 learned 64-d **context tokens** that the cross-attention reads — a miniature of Stable Diffusion's text
cross-attention. Training: MSE between ε̂(x_t, t, y) and ε, class dropped to ∅ 10 % of the time (so the same net is
also an unconditional model → classifier-free guidance).

## 3. Sampling
* **DDPM** (ancestral): x_{t'} = posterior mean(x̂₀, x_t) + σ·z, with σ² = β̃.
* **DDIM** (η = 0, deterministic given the seed): x_{t'} = √ᾱ_{t'} x̂₀ + √(1−ᾱ_{t'}) ε̂.
* **CFG**: ε̂ = ε̂(∅) + g · (ε̂(y) − ε̂(∅)). g = 0 is unconditional, g = 1 conditional, g > 1 over-emphasises the class.
* **Seed**: a 32-bit integer → mulberry32 + Box–Muller. "⚛ QRNG seed" draws the integer from Tsotchke's quantum_rng WASM
  (a classical simulator seeded by `crypto.getRandomValues`, health-tested; replayable by typing the same integer).

## 4. Interpretability instruments (and what each one actually measures)
| instrument | definition | honest limitation |
|---|---|---|
| x̂₀ lens | (x_t − √(1−ᾱ)ε̂)/√ᾱ at every step | it is the model's estimate, not ground truth; at large t it's a blurry mean |
| judge curve | tiny CNN classifier on x̂₀ per step | judge trained on clean data → unreliable on blurry x̂₀ |
| trajectory 3D | PCA of all x_t and x̂₀ vectors to 3D | linear projection; distances in 3D are not faithful |
| activation volume | any recorded layer, channels as depth | magnitudes only |
| attention maps | self-attn (N×N) and cross-attn (N × 4 tokens) at the 4×4 (or 4×8 / 2×2) bottleneck | resolution is tiny by design |
| weight surgery | in-place edits on Float32Arrays: zero, scale, channel ablate, noise, rank-k (Eckart–Young) | exact undo stack; edits persist until reset |
| steering | add α(μ_target − μ_all) to the bottleneck; μ from null-label activations on noised training data at t = T/2 | a single global direction per class |
| activation patching | cache bottleneck of run A (per step, per CFG branch), overwrite in run B inside a t-window | skip connections still carry B's information |
| linear probe | logistic regression on the mean bottleneck at t = T/2 with the ∅ label | trained on null label; on conditional runs it also sees label routing |
| SAE | 64 → 256 ReLU autoencoder on per-position bottleneck vectors, L1 weighted by decoder norms | features are not auto-labelled |

## 5. Modalities (what the tensor *means*)
* **image** 1×16×16 grayscale MNIST digit.
* **music** 1×16×32 piano roll: rows = 2 octaves of C major (row 0 = C4), columns = 1/8-second steps; value > 0 = note on.
  Rendered to sound with a 3-harmonic sine synth. The model never sees audio samples.
* **video** 4×16×16: four frames stacked as channels (so time mixing happens in the first conv; there is no temporal attention).
* **3D** 3×8×8: 64 points (x, y, z in channels), ordered by z-rank (rows) then azimuth (columns) so a CNN sees neighbours.

## 6. Verification
Every claim is backed by a test (`npm test`) or by the pre-registered eval (`node tools/eval.mjs <modality>`). See CLAIMS.md.
