# COURSE — ten labs, each with something to measure

1. **Noise.** SCHEDULE tab. Read ᾱ at t = 199. Question: is x_T pure noise? (Answer: ᾱ_199 ≈ 0.13 — no.)
2. **Sample.** SAMPLE, image, class 7, DDIM 20, g = 3. Press "one per class". Record how many the judge gets right.
3. **Guidance.** Repeat with g = 0, 1, 3, 6. Plot accuracy vs g by hand. Where does quality break?
4. **Schedulers.** DDPM 50 vs DDIM 20 on the same seed. Same digit? (DDPM adds fresh noise each step, so not identical.)
5. **Lens.** LENS tab. At which step does the judge curve lock on to the final class? Compare with PREREG H2.
6. **Trajectory.** TRAJ3D. Do different seeds converge to one region or many? Why can PCA mislead?
7. **Attention.** ATTN. Step through t. Which cross-attention context token gets the most mass? Does it change with t?
8. **Surgery.** SURGERY → `net.ca.o.weight` → zero → "test edited model". Does the class survive without cross-attention? (PREREG H7.) Undo, check parity in METRICS returns ≤ 1e-5.
9. **Steer & patch.** STEER: unconditional + target 3, α sweep. PATCH: A = 1, B = 7; window sweep. Early or late windows matter more? (H3, H4.)
10. **Features.** SAE·PROBE. Find a feature that lights up on stroke ends (image) or on the lower octave (music). Write the claim and a test that could refute it.
