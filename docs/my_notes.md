# My notes — FoundationStereo reproduction & KITTI adaptation

Running log of what we did, why, what we found, and how much to trust it.
Lower is better for every metric. Last updated: 2026-11-17.

**Experiment names (renamed 2026-11-19):** E0 full fine-tune · E1 BN locked · **E2** GRU only (was E4) · **E3a** LoRA all (was E5a) · **E3b** LoRA GRU (was E5b). Files written on Colab/Drive keep their original names (`E4_gru_only.pth`, `E5a_*`, `E5b_*`, `e1_e4_lora_results*`, Drive folder `foundationstereo_e1_e4_lora/`).

---

## 0. Project in one paragraph

Paper: *FoundationStereo: Zero-Shot Stereo Matching* (arXiv 2501.09898, NVlabs). Claim: a stereo model trained on
large synthetic data + frozen DepthAnything-V2 (DINOv2 ViT-L) features generalizes zero-shot to real data.
We (1) reproduce the zero-shot Middlebury number, (2) fine-tune on KITTI to see how much a target domain helps,
(3) measure what fine-tuning costs on a different domain (Middlebury = "retention"), and (4) try to reduce that cost
(BatchNorm locking, partial freezing, LoRA).

## 1. Setup and metric definitions

- Checkpoint: `23-51-11/model_best_bp2.pth` (ViT-L), 32 GRU iterations at test time, mixed precision.
- **Middlebury**: Eval3 trainingH, 15 scenes, half resolution, non-occluded pixels only (`mask0nocc == 255`).
  BP-2 = % pixels with error > 2 px. "Weighted" = pooled over all pixels of all scenes (big scenes count more);
  "mean-scene" = average of the 15 per-scene values.
- **KITTI**: EPE (px) and D1 = % pixels with error > 3 px AND > 5 % of GT. "all" = `disp_occ` GT, "noc" = `disp_noc` GT.
  "Global" D1 = pooled over pixels; "mean-image" D1 = average of per-image values.
- KITTI evaluation is on the public **training** sets (the official test GT is hidden), so it is not the leaderboard metric.

## 2. Zero-shot reproduction (Middlebury) — 2026-11-02

**Why:** confirm our pipeline reproduces the paper before building on it.

| | BP-2 (weighted) | BP-2 (mean-scene) | EPE |
|---|---|---|---|
| Ours (scale 1.0, 32 iters) | **1.0987 %** | 1.1475 % | 0.3634 |
| Paper | ~1.1 % | | |

Source: `data/archives/foundationstereo_middlebury_resultscorrectmetric.zip` (Colab/Kaggle). The same 1.0987 % came out again in
the KITTI notebooks' zero-shot Middlebury evaluation, so it is stable across runs.

**Bugs found on the way**
- `outputs/metrics/middlebury_training_half_fss05_metrics.csv` is WRONG: its `nonocc` and `all` rows are identical
  (Adirondack 1,419,088 px = non-occluded 1,298,347 + occluded 120,741 → mask not applied), and it used an extra 0.5
  downscale (`s05`). Its ~6 % BP-2 must not be used anywhere.
- The first two result zips (`..._results.zip`, `..._results2.zip`) predate the metric fix.

**Critical notes**
- "Matches the paper" depends on aggregation: weighted 1.10 vs mean-scene 1.15. I have not confirmed which one (and
  which mask / resolution) the paper uses, so "reproduced within ~0.05 pts" is the honest statement.

## 3. KITTI protocols

| | Train | Validation (model selection) | Test |
|---|---|---|---|
| Protocol A | KITTI 2012, 194 pairs | KITTI 2015, 40 pairs | KITTI 2015, other 160 pairs |
| Protocol B | KITTI 2012, 155 pairs | KITTI 2012, 39 pairs | KITTI 2015, all 200 pairs |

Split: `np.random.default_rng(2025)`, first permutation over KITTI 2015, second over KITTI 2012. Verified that the
new notebook reproduces the E0 split exactly (val 000139, 000044, 000014, 000146, 000163; test 000129, 000008, 000127, 000102).

**Why two protocols:** A selects the checkpoint with target-domain (KITTI 2015) labels; B never sees KITTI 2015 at all,
so B is the cleaner "cross-dataset" test. Note A and B test sets differ (160 vs 200 pairs) → A vs B is not a strict comparison.

Fine-tuning recipe (all runs): AdamW (wd 1e-5), grad clip 1.0, AMP, batch 1, random 256×512 crops, smooth-L1 sequence
loss (γ = 0.9, 12 train iterations), ReduceLROnPlateau on validation D1-noc (×0.5, patience 2), validation every
50 steps, early stopping after 3 validations without improvement, max 5000 steps. DepthAnything always frozen.

## 4. E0 — full fine-tuning, Protocol A and B (v1 notebook)

**Why:** baseline for "how much does KITTI fine-tuning help, and what does it cost on Middlebury".
Trainable: everything except DepthAnything (39.2 M params). BatchNorm statistics were updating (`model.train()`, batch 1).

### Protocol A (best checkpoint step 350, early-stopped at 500)
Validation curve (D1-noc %): 50: 2.091 · 100: 1.654 · 150: 1.569 · 200: 1.509 · 250: 1.474 · 300: 1.468 ·
**350: 1.442** · 400: 1.449 · 450: 1.501 · 500: 1.467

| | all EPE | all D1 | all mean-img D1 | noc EPE | noc D1 | noc mean-img D1 |
|---|---|---|---|---|---|---|
| Zero-shot val (40) | 0.8745 | 2.7182 | 2.7636 | 0.8661 | 2.6866 | 2.7294 |
| Fine-tuned val | 0.7369 | 1.4720 | 1.4598 | 0.7292 | 1.4421 | 1.4327 |
| Zero-shot test (160) | 0.8545 | 2.8561 | 2.9054 | 0.8351 | 2.7549 | 2.8204 |
| Fine-tuned test | 0.7343 | 1.8623 | 1.7978 | 0.7152 | 1.7146 | 1.6744 |

Middlebury: zero-shot BP-2 1.0987 % / EPE 0.3634 → fine-tuned **BP-2 2.4225 % / EPE 0.6345**.

### Protocol B
| | all EPE | all D1 | all mean-img D1 | noc EPE | noc D1 | noc mean-img D1 |
|---|---|---|---|---|---|---|
| Zero-shot val (39, KITTI 2012) | 0.5897 | 2.1502 | 2.2162 | 0.5487 | 1.8960 | 1.9545 |
| Fine-tuned val | 0.4789 | 1.3138 | 1.3483 | 0.4055 | 0.8921 | 0.9149 |
| Zero-shot test (200, KITTI 2015) | 0.8583 | 2.8307 | 2.8774 | 0.8410 | 2.7422 | 2.8022 |
| Fine-tuned test | 0.7319 | 1.7005 | 1.6484 | 0.7155 | 1.5948 | 1.5613 |

Middlebury: fine-tuned **BP-2 2.4036 % / EPE 0.6514**.

### E0 findings
- KITTI 2015 D1-all: −34.8 % (A), −39.9 % (B) vs zero-shot. EPE only −14 %: fine-tuning mostly removes large errors.
- Middlebury BP-2 more than doubles in both protocols (+120 %, +119 %); EPE +75–79 %.
- Zero-shot KITTI 2015 D1 (2.83–2.86 %) agrees with the paper's 2.8 %.

### E0 critical notes
- Protocol A was run twice with the same seed: test D1-all 1.8715 vs 1.8623, best step 350 both times. Same-seed
  GPU non-determinism is therefore small (~0.01 pts). **Different-seed variance was never measured.**
- B beating A on KITTI 2015 (1.70 vs 1.86) is NOT evidence that B is better: different test sets, different seeds (2025 vs 2026),
  different validation sets. Treat as "comparable".
- The ~2× Middlebury degradation is the robust result: same size in A and B.
- Per-image rows for E0 Protocol A were not saved (only aggregates) → E0 cannot be compared image-by-image with later runs.

## 5. Architecture notes (what can be frozen)

`feature` = frozen DepthAnything-V2 (`feature.dino`) + trainable EdgeNeXt CNN + fusion (paper's Side-Tuning Adapter) →
cost volume (`proj_cmb`, `corr_stem`) → cost filtering (`corr_feature_att`, `cost_agg` hourglass + disparity transformer,
`classifier` → initial disparity) → context net (`cnet`, `cam`, `sam`) → ConvGRU refinement (`update_block`, 12/32 iters)
→ learned upsampling (`stem_2`, `spx_2_gru`, `spx_gru`).
BatchNorm lives in `cnet`, `corr_stem`, `cost_agg`, `classifier` → with batch size 1 in `train()` mode its running
statistics drift toward KITTI even where weights barely move. This motivated E1.

## 6. E1 / E2 / E3 — reducing the Middlebury cost (Protocol A only) — 2026-11-17

Notebook: `colab_e1_e3_lora_bundle/foundationstereo_e1_e3_lora.ipynb`. Results: `experiments/03_retention_E1_E3/results/run1/` (extracted from `experiments\03_retention_E1_E3\results\run1\` (extracted from `e1_e4_lora_results (1).zip`)).
Same recipe and same crop sequence as E0; additions: BatchNorm statistics locked in all runs, validation at step 0,
best weights start from the pretrained model. LoRA: rank 8, alpha 16, LR 1e-4 (others 1e-5).

| Run | Why | Trainable |
|---|---|---|
| E1_bn_locked | Is BatchNorm drift part of the Middlebury loss? | all except DepthAnything (39.2 M) |
| E2_gru_only | Can refinement alone adapt? Where should LoRA go? | GRU + upsampler (16.4 M) |
| E3a_lora_all | Low-rank version of E1 | LoRA on 217 layers (1.86 M) |
| E3b_lora_gru | Low-rank version of E2 | LoRA on 40 GRU/upsampler layers (0.67 M) |

### Results (KITTI 2015 held-out test, 160 pairs; Middlebury 15 scenes)

| Model | Params | Best / stop step | Val D1-noc | Test D1-all | Test D1-noc | Test EPE | MB BP-2 | MB mean-scene BP-2 | MB EPE |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0 | – | 2.6866 | 2.8561 | 2.7549 | 0.8545 | 1.0987 | 1.1475 | 0.3634 |
| E0 full FT | 39.2 M | 350 / 500 | 1.4421 | 1.8623 | 1.7146 | 0.7343 | 2.4225 | – | 0.6345 |
| E1 BN locked | 39.2 M | 1000 / 1150 | 1.3173 | **1.4979** | **1.3625** | 0.7145 | **2.1074** | 2.2387 | 0.6552 |
| E2 GRU only | 16.4 M | 600 / 750 | 1.4524 | 1.6191 | 1.4813 | 0.7234 | 2.2676 | 2.4039 | 0.7359 |
| E3a LoRA all | 1.86 M | 350 / 500 | 1.3818 | 1.5747 | 1.4515 | **0.7142** | 2.1765 | 2.3104 | 0.6674 |
| E3b LoRA GRU | 0.67 M | 950 / 1100 | 1.4177 | 1.5752 | 1.4428 | 0.7180 | 2.3446 | 2.5048 | 0.7260 |

Change vs zero-shot: KITTI test D1-all E0 −34.8 %, E1 −47.6 %, E2 −43.3 %, E3a −44.9 %, E3b −44.8 %;
Middlebury BP-2 E0 +120 %, E1 +92 %, E2 +106 %, E3a +98 %, E3b +113 %.

Training: E1 73 min (LR halved at 550 and 1150), E2 41 min (LR halved at 750), E3a 40 min (at 500), E3b 69 min (at 1100).

Validation curves (D1-noc %):
- E1: 0: 2.687 · 50: 1.781 · 200: 1.478 · 350: 1.452 · 400: 1.421 · 450: 1.420 · 600: 1.395 · 700: 1.336 · 850: 1.319 · **1000: 1.317** · 1150: 1.325
- E2: 0: 2.687 · 50: 2.098 · 200: 1.609 · 400: 1.534 · 550: 1.472 · **600: 1.452** · 750: 1.470
- E3a: 0: 2.687 · 50: 1.936 · 200: 1.453 · 300: 1.413 · **350: 1.380** · 500: 1.429
- E3b: 0: 2.687 · 50: 2.256 · 200: 1.568 · 500: 1.519 · 800: 1.437 · **950: 1.418** · 1100: 1.474

Middlebury BP-2 per scene (%):

| Scene | Zero-shot | E1 | E2 | E3a | E3b |
|---|---|---|---|---|---|
| ArtL | 1.13 | 4.01 | 4.03 | 4.13 | 4.59 |
| Pipes | 1.56 | 3.94 | 4.42 | 4.10 | 3.98 |
| Jadeplant | 3.70 | 5.96 | 7.70 | 6.16 | 7.24 |
| PianoL | 0.56 | 1.95 | 1.44 | 2.20 | 1.87 |
| Piano | 0.51 | 1.71 | 1.48 | 1.47 | 1.61 |
| Shelves | 0.64 | 1.84 | 1.61 | 1.40 | 1.61 |
| Motorcycle | 1.41 | 2.55 | 2.96 | 2.85 | 3.01 |
| MotorcycleE | 1.44 | 2.36 | 2.67 | 2.62 | 2.76 |
| PlaytableP | 0.98 | 1.80 | 1.96 | 1.90 | 2.29 |
| Playtable | 0.96 | 1.74 | 1.85 | 1.92 | 2.26 |
| Recycle | 0.51 | 1.20 | 1.24 | 1.22 | 1.28 |
| Adirondack | 0.28 | 0.71 | 0.72 | 0.74 | 0.83 |
| Vintage | 0.50 | 0.71 | 0.71 | 0.72 | 0.70 |
| Teddy | 1.31 | 1.50 | 1.45 | 1.49 | 1.60 |
| Playroom | 1.73 | 1.60 | 1.81 | 1.73 | 1.96 |

### Statistical checks (paired bootstrap, 10,000 resamples)
These only capture image/scene sampling noise, NOT seed-to-seed training noise (one seed per run).

KITTI test D1-all, 160 images (A − B, negative = A better):
| Comparison | Δ pts | 95 % CI | images where A better |
|---|---|---|---|
| E1 − E3a | −0.077 | [−0.132, −0.023] | 120 / 157 |
| E1 − E2 | −0.121 | [−0.187, −0.074] | 119 / 159 |
| E1 − E3b | −0.077 | [−0.116, −0.035] | 117 / 159 |
| E3a − E2 | −0.044 | [−0.085, −0.008] | 90 / 160 |
| E3b − E2 | −0.044 | [−0.136, +0.011] | 92 / 157 |
| E3a − E3b | −0.001 | [−0.066, +0.086] | 66 / 159 |

Middlebury BP-2, 15 scenes:
| Comparison | Δ pts | 95 % CI |
|---|---|---|
| E1 − E3a | −0.069 | [−0.166, +0.046] |
| E1 − E2 | −0.160 | [−0.431, +0.054] |
| E1 − E3b | −0.237 | [−0.431, −0.073] |
| E3a − E3b | −0.168 | [−0.332, −0.034] |

Every run is worse than zero-shot on 14–15 of 15 Middlebury scenes; BP-2 ratio vs zero-shot 95 % CI:
E1 [1.61, 2.33], E2 [1.74, 2.38], E3a [1.69, 2.38], E3b [1.85, 2.48].

Sanity checks passed: Middlebury valid-pixel counts identical across all runs and equal to the zero-shot CSV;
step-0 validation of E1 (2.6866) equals the zero-shot reference exactly (E2/E3: 2.687, i.e. ~0.0004 eval non-determinism).

### Findings (with confidence)
1. **Middlebury degradation is robust and not fixed by any method** (high confidence). All runs 1.9–2.1× zero-shot BP-2,
   CIs far above 1. Locking BN, freezing, and LoRA all leave most of the damage.
2. **Restricting what is trained does not protect Middlebury** (high confidence). 0.67 M LoRA params on the GRU (E3b)
   still more than double BP-2, and is the worst new run on Middlebury (significantly worse than E1 and E3a).
   GRU-only runs (E2, E3b) have the worst Middlebury EPE (0.73–0.74). → The refinement stage alone is enough to
   absorb KITTI-specific behaviour.
3. **E1 is the best KITTI model** (moderate–high confidence for this seed): significantly better than every other new run
   on KITTI (all CIs exclude 0, ~75 % of images better).
   **⚠ REVISED in 6b:** in the same-seed rerun E1 stopped at step 400 and is no longer significantly better than E3a/E3b.
4. **E1 vs E0 (BN locking effect) — suggestive, NOT clean** (low–moderate confidence):
   - KITTI 1.86 → 1.50 and Middlebury BP-2 2.42 → 2.11 look large, but E1 also trained ~3× longer (best step 1000 vs 350).
     Up to step 350 the curves were nearly identical (E1 1.452 vs E0 1.442); E0 stopped at 500 after a validation bump
     at 450 (1.501) that E1 did not have. So most of the KITTI gain is "trained longer", not clearly BN.
   - Middlebury BP-2 improved while EPE got slightly worse (0.635 → 0.655): fewer bad pixels, larger errors on the rest.
   - No E0 per-image rows → no paired test possible. Different stopping points → confounded.
5. **LoRA-all (E3a) is very parameter-efficient on KITTI** (moderate): 1.57 % with 21× fewer params than E1,
   best test EPE, but still significantly worse than E1 on D1 (−0.077, CI excludes 0). Not better than E1 on Middlebury
   (difference within noise).
6. **E3a vs E3b are tied on KITTI** (CI spans 0), but E3a is clearly better on Middlebury. Putting LoRA everywhere is
   better than only in the GRU.
7. **Where Middlebury breaks:** thin-structure / fine-detail scenes (ArtL ×3.5, Pipes ×2.5, Jadeplant, Piano/PianoL,
   Shelves). Large planar / textured scenes barely change (Vintage, Teddy; Playroom slightly better).
   *Hypothesis (unverified):* KITTI's sparse LiDAR GT has little supervision on thin structures, so fine-tuning learns to
   smooth them. Needs error maps or a thin-structure mask to confirm.

### Critical caveats for everything in section 6
- **One seed per configuration.** Same-seed noise is small (~0.01 pts) but seed-to-seed noise is unknown and is
  probably larger than some KITTI gaps here (0.04–0.12 pts). Rankings among E1/E2/E3a/E3b on KITTI need ≥ 2 more seeds.
- **Early stopping is noisy.** Validation D1 moves 0.02–0.05 pts between consecutive evaluations on 40 images, and
  best steps ranged 350–1000. Where a run stops is partly luck (see E0 vs E1).
- **Training loss curves are almost identical across runs at the same steps** (e.g. spikes at steps 600 and 750 in every run):
  loss is dominated by which crop is drawn (same crop sequence), so it says little about the method.
- **Concentration:** in every run the worst 10 of 160 test images hold ~29 % of all D1 bad pixels (max image D1 ≈ 20 %).
  Aggregate D1 is sensitive to a few hard images.
- **Validation is optimistic** (selection on it): val D1-all 1.34 vs test 1.50 for E1.
- **Repeated use of evaluation sets:** configurations were chosen after seeing E0's Middlebury result, and the same 160 KITTI
  test images / 15 Middlebury scenes were reused for every run. There is no untouched Middlebury hold-out; mild selection bias.
- **LoRA used LR 1e-4 vs 1e-5** for full fine-tuning: E1 vs E3a compares "method + LR", not method alone.
- E1/E2/E3 compare against E0 numbers from an earlier notebook (same split and metric code); zero-shot numbers were not
  recomputed in the new notebook except the step-0 validation, which matched.
- 15 Middlebury scenes is a small sample; CIs over scenes are wide (e.g. E1 − E2 Middlebury CI [−0.43, +0.05]).

## 6b. Second run of the same notebook (Drive export) — analysed 2026-11-17
**What:** the user exported `MyDrive/FoundationStereo_Training` as 4 zips (parts -001, -003, -004, -005; part -002
missing — probably the 3.3 GB base model). They contain full checkpoints E1/E2/E3a/E3b, LoRA adapters, results JSON,
rows, comparison CSV. **These are a second execution of the E1/E2/LoRA notebook (same seed 2025, same split, same code)**,
which overwrote the run-1 checkpoints on Drive. Run-1 metrics (section 6) came from `experiments/03_retention_E1_E3/results/run1/` (extracted from `experiments\03_retention_E1_E3\results\run1\` (extracted from `e1_e4_lora_results (1).zip`)).
Evidence: E1 best step 400 (run 1: 1000); full E3 checkpoints match the Drive adapters to 1e-8 but differ from the
run-1 adapters by ~3e-3.

**Run 1 vs run 2 (Protocol A; KITTI 2015 test 160 img, Middlebury trainingH 15 scenes)**

| Run | | best/stop | val D1-noc | test D1-all | test D1-noc | test EPE | MB BP-2 (w) | MB BP-2 (mean) | MB EPE |
|---|---|---|---|---|---|---|---|---|---|
| E1  | run1 | 1000/1150 | 1.3173 | 1.4979 | 1.3625 | 0.7145 | 2.1074 | 2.2387 | 0.6552 |
| E1  | run2 | 400/550   | 1.4131 | **1.5444** | 1.4196 | 0.7170 | **2.0831** | 2.2117 | 0.6629 |
| E2  | run1 | 600/750   | 1.4524 | 1.6191 | 1.4813 | 0.7234 | 2.2676 | 2.4039 | 0.7359 |
| E2  | run2 | 600/750   | 1.4516 | 1.6187 | 1.4815 | 0.7232 | 2.2700 | 2.4063 | 0.7376 |
| E3a | run1 | 350/500   | 1.3804 | 1.5747 | 1.4515 | 0.7142 | 2.1765 | 2.3104 | 0.6674 |
| E3a | run2 | 350/500   | 1.3926 | 1.5896 | 1.4616 | 0.7159 | **2.1082** | 2.2389 | 0.6499 |
| E3b | run1 | 950/1100  | 1.4180 | 1.5752 | 1.4428 | 0.7180 | 2.3446 | 2.5048 | 0.7260 |
| E3b | run2 | 950/1100  | 1.4181 | 1.5755 | 1.4431 | 0.7175 | 2.3239 | 2.4824 | 0.7211 |

**Findings (this is the first real measurement of run-to-run noise)**
1. **GPU non-determinism is not negligible.** Same seed/code, yet 91–99 % of test images change. Validation curves
   drift by up to ~0.03 pts (E1 step 300: 1.4574 vs 1.4300; E3a step 450: 1.4065 vs 1.4300).
2. **Early stopping amplifies it — E1 is the clear case.** Run 1: step 450 = 1.4203 < 1.4209 (best at 400) → counted as
   improvement, patience reset, training continued to 1150 and reached 1.3173. Run 2: step 450 = 1.4138 > 1.4131 →
   no improvement, stopped at 550. A 0.0006 difference at one evaluation changed the training length ×2.9 and test
   D1-all by +0.047 pts. **E1's run-1 result was partly early-stopping luck.**
3. **The run-1 claim "E1 significantly beats every other run on KITTI" does NOT replicate.** Run-2 paired bootstrap
   (KITTI test D1-all, 10k resamples over images):
   - E1 − E3a = −0.045, 95 % CI [−0.135, +0.023] → n.s. (run 1: significant)
   - E1 − E3b = −0.031, CI [−0.077, +0.019] → n.s. (run 1: significant)
   - E1 − E2 = −0.074, CI [−0.166, −0.001] → barely significant
   - E3a − E3b = +0.014, CI [−0.055, +0.116] → n.s. (tie, same as run 1)
   - E3b − E2 = −0.043, CI [−0.133, +0.009] → n.s.
   Ranking E1 < E3b ≈ E3a < E2 is stable in sign, but gaps between E1/E3a/E3b are within run-to-run noise.
4. **Middlebury run-to-run noise:** weighted BP-2 moved −0.02 to −0.07 pts (E3a 2.18 → 2.11; per-scene up to ±0.47 for E1
   PianoL). Differences < ~0.1 pts between methods on Middlebury should not be interpreted.
5. **Middlebury damage happens early.** E1 trained 400 steps (run 2) vs 1000 (run 1) and Middlebury BP-2 is the same
   (2.08 vs 2.11). So "train less" does not protect retention; the ~2× degradation appears within the first few hundred
   steps. (Consistent with E0: 350 steps, 2.42.)
6. **Robust across both runs:** every fine-tuned model ≈ 1.9–2.1× zero-shot Middlebury BP-2; E2 is worst on KITTI;
   E3b ≈ E2 on Middlebury (worst); GRU-only and LoRA-GRU runs are deterministic-ish (smallest drift), full/LoRA-all drift more.
7. Runs E2 and E3b are nearly reproducible (|ΔD1| ≤ 0.001), suggesting the non-determinism comes mostly from
   training the earlier (feature/cost-volume) layers.

**Checkpoint verification (CPU tensor comparison, no inference):**
- E1: DepthAnything (`feature.dino`) unchanged vs base ✔; E2: only `update_block`, `spx_2_gru`, `spx_gru`, `stem_2`
  parameters changed ✔; E3a/E3b full checkpoints = base + Drive adapter merged (max |diff| 1.5e-8) ✔.
- **Base-model mismatch:** in every Colab checkpoint, 12 BatchNorm buffers (`cnet.layer2–5.0.norm3` running_mean/var/
  num_batches_tracked) differ from our local `model_best_bp2.pth` — identically in E1, E2, E3a, E3b (num_batches_tracked
  200001 vs 196541 locally). This is NOT BN drift (BN locking worked); the Hugging Face mirror
  (`steve-redefine/FoundationStereoWeights`) used on Colab is a slightly different snapshot from our local official file
  in these 12 buffers. Effect on results is presumably tiny but unverified. Rebuilding E3 from local base + adapter therefore
  differs from the Colab model in these buffers → use the full E3 checkpoints for inference (now available).
- Run-1 full checkpoints no longer exist (overwritten). Only the run-1 LoRA adapters survive
  (`inference/models/archive_run1/`).

## 6c. Classical baseline (SGBM), run locally on CPU 2026-11-19
**Why:** a non-learned reference point for the slides ("how far is FoundationStereo from classical stereo?").
Script `inference/classical_baseline.py` (commands: `inference/RUN_COMMANDS.txt` section 13). OpenCV StereoSGBM, settings
fixed in advance (block 5, P1 8·3·25, P2 32·3·25, uniqueness 10, speckle 100/2, MODE_SGBM; ndisp = Middlebury calib.txt
value, KITTI 256). Same evaluation code as the notebooks: Middlebury trainingH 15 scenes nocc BP-2/EPE pixel-weighted;
KITTI 2015 Protocol A test 160 images (rng 2025, split checked against E0), D1/EPE all + noc. Rejected pixels filled by
KITTI-devkit background interpolation; "density" (% matched before filling) is reported. Optional plain BM for illustration.
- Results (`inference/outputs/classical_baseline/sgbm/summary.md`; KITTI 2015 downloaded to `data/kitti/raw/kitti2015`,
  only image_2/image_3/disp_occ_0/disp_noc_0 `_10` frames, zip deleted; split check vs E0 passed):

  | | MB BP-2 (w) | MB BP-2 (mean) | MB EPE | KITTI test D1-all | D1-noc | mean-img D1-all | KITTI EPE all |
  |---|---|---|---|---|---|---|---|
  | SGBM | 23.5551 | 23.0047 | 5.6420 | 18.0544 | 17.0770 | 17.5870 | 2.8670 |
  | BM (plain block matching) | 29.0340 | 28.6048 | 8.6198 | 23.1891 | 22.1557 | 22.9577 | 4.2338 |
  | Zero-shot FS | 1.0987 | 1.1475 | 0.3634 | 2.8561 | 2.7549 | 2.9054 | 0.8545 |
  | E1 (run 1) | 2.1074 | 2.2387 | 0.6552 | 1.4979 | 1.3625 | – | 0.7145 |

  SGBM ≈ 21× zero-shot BP-2 on Middlebury and ≈ 6.3× zero-shot D1 on KITTI; every fine-tuned FS model is still ~10× better
  than SGBM on Middlebury. Matched-pixel density: Middlebury 54–83 % per scene, KITTI ~56–72 % per image (left border
  without match counts as unmatched). Worst MB scenes: Vintage 46 %, PianoL 42 % (lighting change), Shelves 38 %,
  Jadeplant 32 %; best Motorcycle 7 %. Speed ~1.5 s / Middlebury scene, ~0.6 s / KITTI image on CPU.
- BM (OpenCV StereoBM, grayscale, block 15, same holes rule) run 2026-11-19 on CPU: matches only 33–68 % of Middlebury
  pixels; worse than SGBM on both datasets, as expected without the smoothness term.
- Slide 17 (trade-off) now has two panels: log-scale overview with BM + SGBM, linear zoom on learned models.
- On the slides: slide 11 "Classical stereo (SGBM) is 6–21× worse" (Adirondack GT | SGBM | FS + bad-pixel maps).
- Caveat: SGBM uses a fixed setting; a tuned classical method would do better, so this is a lower reference, not the
  best classical number.

## 7. Open questions / next steps
- [x] ~~Same-seed rerun~~ — done by accident (section 6b): run-to-run noise ≈ 0.01–0.05 pts KITTI, 0.02–0.07 pts Middlebury;
  early stopping can flip a run.
- [ ] Make early stopping less fragile: require improvement > threshold (e.g. 0.005) or smooth over 2–3 evals; or fix
  the step budget and report the last checkpoint.
- [ ] Add 2 more seeds for E1 and E3a (and ideally E0) → seed variance; decide whether rankings hold.
- [ ] Re-run E0 with BN locked vs unlocked, both stopped at the same step (e.g. 350 and 1000) → clean BN effect.
- [ ] Save E0 per-image rows so E0 can be compared pairwise.
- [ ] Middlebury error maps for ArtL / Pipes / Jadeplant (zero-shot vs E1) → test the thin-structure hypothesis.
- [ ] Middlebury error binned by GT disparity (KITTI disparities are small) → check for a disparity-range prior.
- [ ] Weight averaging (WiSE-FT) between zero-shot and E1: free, may trade KITTI vs Middlebury smoothly.
- [ ] ETH3D as a second retention dataset.
- [ ] Protocol B for the best new configuration (E1) to check it carries over.

## 8. File map
**Repository layout since 2026-11-20 (prepared for GitHub, https://github.com/Murfing/FoundationStereo_finetuning):**
`FoundationStereo/` (official code + patch) · `experiments/01_middlebury_zero_shot`, `02_kitti_finetuning_E0`,
`03_retention_E1_E3` (notebooks + results; one shared `foundationstereo_source_bundle.zip`) · `inference/` (scripts,
models.json, BM/SGBM outputs, demo pairs) · `data/` (KITTI, Middlebury, archives; git-ignored) · `docs/` (these notes,
presentation notes; `docs/local_only/` git-ignored). The old `Project/` folder was dissolved: duplicate code copies and
3 duplicate source zips removed; run-1 results zip extracted to `experiments/03_retention_E1_E3/results/run1/` (its
adapters are identical to `inference/models/archive_run1/`). Entries below may still use the old paths.

- Paper: `docs/local_only/foundationstereo_paper.pdf`; official code: `FoundationStereo/`
- Zero-shot Middlebury: `data/archives/foundationstereo_middlebury_resultscorrectmetric.zip`
- E0 Protocol A: `experiments/02_kitti_finetuning_E0/results/protocol_a_summary.json`
- E0 Protocol B: `experiments/02_kitti_finetuning_E0/results/protocol_b_summary.json` (per-image rows),
  checkpoint `inference/models/E0_B_full_ft_protocol_b.pth`
- Combined E0 results: `experiments/02_kitti_finetuning_E0/results/final_results_E0_A_B.json`
- v1 notebook (working): `experiments/02_kitti_finetuning_E0/foundationstereo_kitti_finetuning.ipynb`
- E1/E2/E3 notebook + bundle: `experiments/03_retention_E1_E3/`; results `experiments/03_retention_E1_E3/results/run1/` (extracted from `experiments\03_retention_E1_E3\results\run1\` (extracted from `e1_e4_lora_results (1).zip`));
  full checkpoints in Google Drive `MyDrive/foundationstereo_e1_e4_lora/checkpoints/`
- Inference for any model (2026-11-17): `inference/infer.py` + `inference/models.json` (model registry);
  `--model zero_shot,E0_A,E0_B,E1,E2,E3a,E3b|all`; same eval conventions (32 iters, pad /32, fp16, D1 >3px & >5%).
  `inference/models/` holds zero_shot_foundationstereo.pth (+cfg.yaml, local official file), E0_B_full_ft_protocol_b.pth
  (step 700, val D1-noc 0.892), and from the Drive export (run 2, section 6b): E1_bn_locked.pth, E2_gru_only.pth,
  E3a_lora_all.pth, E3b_lora_gru.pth + matching *_adapter.pth; run-1 adapters in models/archive_run1/.
  E0_A is not available anywhere (was only in Colab /content). Run-2 metrics/rows:
  `experiments/03_retention_E1_E3/results/run2/`. Commands: `inference/RUN_COMMANDS.txt`; Colab: `inference/foundationstereo_inference_colab.ipynb`.
- Local inference check (2026-11-17, GTX 1650, zero_shot, `--fp32 --scale 0.5`, iters 32):
  Adirondack (Middlebury H, nocc mask) EPE 0.325, D1 0.98%, BP-2 **1.34%** vs **0.28%** in the full-res fp16 Colab
  reproduction -> ~5x worse purely from halving input resolution. Local scale<1 numbers are NOT comparable to the
  project metrics. Timing: checkpoint load 18.5 s; 960x540 demo 5.4 s; 1436x992 (run at 718x496) 16.7 s.
- Presentation (2026-11-19): slide deck https://claude.ai/artifact/UdwYKGCB7M4eTgKcfEHDFC (19 slides, paper figures
  cropped from the PDF, our numbers from this file); speaker notes + likely Q&A in `docs/presentation.md`.
- Deleted: `colab_kitti_bundle_v2/` (its KITTI 2015 zip was corrupted: 2,998 / 3,415 entries unreadable)
- Cleanup 2026-11-17 (all verified redundant before deleting; ~11 GB freed):
  4 Drive-export zips (contents extracted + size-checked); `Downloads/protocol_b_finetuned.pth` and
  `Project/checkpoints/foundationstereo_pretrained/` (byte-identical copies now in `inference/models/`);
  `foundationstereo_middlebury_results.zip` (same predictions as correctmetric, wrong metrics) and `_results2.zip`
  (pre-metric-fix run); Middlebury data/GT zips (fully extracted in `data/middlebury/raw/`);
  `Project/outputs/middlebury_one_shot/` + `outputs/metrics/..._fss05_metrics.csv` (old scale-0.5 local run, wrong mask);
  `foundationstereo_e1_e2_lora_bundle.zip` (= notebook + source zip); `foundationstereo_kitti_finetuning_v2.ipynb`
  (byte-identical to v1); `__pycache__`.
- 2026-11-18: removed `FoundationStereo/.git` (upstream NVlabs/FoundationStereo @ 6e88068) and the unrelated Splashtop
  installer. The only local code change vs upstream (`depth_anything/dpt.py`: load DINOv2 from the bundled `dinov2/`
  folder first, fall back to torch.hub online) is saved in `FoundationStereo/LOCAL_CHANGES_vs_upstream_6e88068.patch`.
