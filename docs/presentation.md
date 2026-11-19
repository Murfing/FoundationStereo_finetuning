# FoundationStereo presentation: speaker notes

Deck: https://claude.ai/artifact/UdwYKGCB7M4eTgKcfEHDFC (21 slides; Present mode shows these notes too).
Target length: ~18–20 minutes + questions. Presenters: Archit Shukla (CH23B072) and Akshit Singh (CH23B071) (instructor: Prof. Sukhendu Das). Suggested split: one speaker takes
slides 1–11 (paper, reproduction, classical baseline), the other slides 12–21 (fine-tuning experiments, findings, demo).

**Flow:** paper (problem → dataset → architecture → results) → we reproduce it → we fine-tune it on KITTI → it forgets Middlebury → we try 4 ways to
prevent forgetting → none does → findings, demo, next steps.

Every number in the deck is from `docs/my_notes.md` (our runs) or the paper PDF (`docs/local_only/foundationstereo_paper.pdf`).
Paper figures were cropped from the PDF: Fig. 1, 2, 3 (right), 4, 7; Table 2 re-typed (selected rows).

---

## Part 1: The paper (slides 1–8, ~7 min)

### 1. Title (~0:30)
Introduce yourselves (Archit Shukla, Akshit Singh) and thank Prof. Sukhendu Das. One-line pitch: we took NVIDIA’s FoundationStereo, a stereo model built to work on any scene
without retraining, reproduced its headline zero-shot numbers, and asked a practical question: if you fine-tune it on a
driving dataset (KITTI), how much do you gain there and how much general ability do you lose elsewhere (Middlebury)?
We report everything honestly, including what did not work.

### 2. Contents (~0:30)
Five parts: (1) the paper: problem, dataset, architecture, reported results; (2) our reproduction plus classical
baselines; (3) fine-tuning on KITTI and its cost on Middlebury; (4) four methods to keep generality and the trade-off;
(5) findings, phone-photo demo, questions.

### 3. Problem (~1:00)
Stereo matching: given two rectified images, find for each pixel how far it moved horizontally: the disparity. Near
objects shift more. With calibration, depth Z = f·B/d. Most strong stereo networks only win benchmarks after
fine-tuning on each benchmark’s own training data. The paper’s claim: one model that works zero-shot on indoor, outdoor,
reflective, thin and textureless scenes. Figure 1 from the paper: no fine-tuning, and the output is good enough for a
metric point cloud.

### 4. FSD dataset (~1:00)
We start with the data. Built by NVIDIA, not by us; we use their released model trained on it. 1M pairs at 1280×720, path-traced in Omniverse
(Scene Flow has ~40K). Baseline and focal length both vary → wide disparity range. Two styles: realistic (indoor/outdoor,
physically settled objects; navigation, driving, manipulation) and chaotic flying objects with random textures that force
true matching. Bottom image: scene models incl. a real vs simulated warehouse. 48 A40 GPUs × 10 days.

### 5. Self-curation (~0:45)
Randomization creates some impossible pairs (perfect repeated texture, mirrors without context). Loop: train, evaluate on
FSD itself, discard pairs with BP-2 > 60 %, regenerate; done twice. Supplementary Table 8: 1.15 with vs 1.27 without, same size. Next: the network trained on this data.

### 6. Architecture overview (~1:15)
Paper Figure 2, left to right.
- **Features:** each image goes through a frozen DepthAnything V2 (monocular depth model trained on internet-scale real
  data) plus a small trainable CNN (EdgeNeXt-S). That pairing is the Side-Tuning Adapter (STA). A separate context
  network sees the left image only.
- **Cost filtering:** left and right features form a 4D cost volume (one slice per candidate disparity), filtered by AHCF.
- **Refinement:** soft-argmin gives an initial disparity at 1/4 resolution; ConvGRU blocks refine it iteratively, looking up
  the cost volume at the current estimate each time.
Key for us: which parts exist, because later we freeze some and train others.

### 7. Components and ablations (~1:15)
Numbers are Middlebury BP-2 from the paper’s ablations (trained on a 100K subset, so higher than the final 1.1).
- STA: DepthAnything V2-L beats DINOv2-L (1.97 vs 2.46). **Keeping the ViT frozen matters: unfreezing gives 3.94, double.**
  Foreshadows our result.
- AHCF: axial-planar convolution splits a 3D conv into spatial (3×3×1) and disparity (1×1×17) parts so a long disparity
  kernel is affordable (17: 1.97 vs 5: 2.10). Disparity Transformer attends along disparity only (1.97 vs full volume 2.25).
- Refinement: ConvGRU, sequence loss with γ = 0.9 weighting later iterations more.
- Picture: without STA the lamp and guitar hole are wrong; without AHCF the repetitive cobblestones are wrong.

### 8. Reported results (~1:00)
Paper Table 2. Last row: Middlebury BP-2 1.1 %, ETH3D BP-1 0.5 %, KITTI-12 D1 2.3 %, KITTI-15 D1 2.8 %, none seen in
training. Even the Scene-Flow-only variant (blue row) beats all competitors → the architecture helps, not only the data.
Metric definitions on the right; we use Middlebury BP-2 and KITTI-15 D1 from here on.

---

## Part 2: Reproduction and classical baseline (slides 9–11, ~3 min)

### 9. We reproduce the numbers (~1:15)
Released checkpoint, half-resolution Middlebury, 15 training scenes, non-occluded pixels, 32 iterations: **1.0987 %**,
matching the paper’s 1.1; identical value again inside the KITTI notebooks → stable. Caveat: per-scene averaging gives 1.15
and the paper doesn’t say which aggregation it uses → “reproduced within ~0.05 pts”. KITTI-15: 2.83–2.86 % (200 or 160
images) vs 2.8. KITTI evaluated on public training images (test labels hidden). Lesson: an early evaluation of ours gave
~6 % because the occlusion mask wasn’t applied (and input was downscaled). Evaluation details matter.

### 10. Our outputs (~0:45)
Adirondack: crisp chair arms, mug, books; 0.28 % BP-2 at full resolution on Colab. On a 4 GB laptop GPU we had to halve
resolution → 1.34 %, ~5× worse. Resolution alone changes results, so all reported numbers are full resolution.
Bottom: repo demo pair. Warm = near, cool = far.

---

### 11. Classical baseline: SGBM (~1:00)
Reference point without learning: OpenCV Semi-Global Block Matching. Idea: match 5×5 windows along the same row, then
penalise disparity jumps between neighbours along 5 directions (P1/P2) so surfaces come out smooth. Standard settings
fixed in advance, not tuned on test data; unmatched pixels (17–46 % per Middlebury scene) filled with the KITTI devkit rule
and scored like FoundationStereo. Same test sets as all our runs (15 Middlebury scenes, 160 KITTI-15 Protocol A images).
- Middlebury BP-2 23.56 % vs 1.10 % zero-shot (~21× worse); KITTI-15 D1-all 18.05 % vs 2.86 % (~6× worse).
- Table also lists plain BM (no smoothness step): 29.03 % / 23.19 %, worse than SGBM, as expected.
- Picture (Adirondack): SGBM fails on the textureless grey chair, the left border (no match possible) and leaves
  horizontal streaks; FoundationStereo errors are thin object edges only. Per-image: SGBM 20.2 % vs FS 1.34 % (laptop run
  at half input scale, so FS would be even better at full resolution: 0.28 %).
- Takeaway: even E1, which doubled its Middlebury error after KITTI tuning (2.11 %), is ~10× better than SGBM.

## Part 3: Fine-tuning on KITTI (slides 12–13, ~2.5 min)

### 12. Protocols (~1:00)
- A: train 194 KITTI-12, choose checkpoint on 40 KITTI-15, test on the other 160.
- B: train 155 KITTI-12, validate on 39 KITTI-12, test on all 200 KITTI-15 (never sees KITTI-15 before test).
- Everything also scored on Middlebury = retention. Same recipe: AdamW, lr 1e-5, batch 1, 256×512 crops, early stopping,
  DepthAnything frozen (as the paper recommends).

### 13. E0 full fine-tuning (~1:30)
KITTI-15 D1: 2.86 → 1.86 (A, −35 %), 2.83 → 1.70 (B, −40 %). Middlebury BP-2: 1.10 → 2.42 / 2.40 (+120 %).
A vs B: B looks better on KITTI but test sets differ (160 vs 200 images) → no claim. EPE only −14 %: fine-tuning mostly
removes large errors. Robust: specialization costs ~2× on Middlebury. That motivated part 4.

---

## Part 4: Keeping generality (slides 14–18, ~5.5 min)

### 14. Four methods (~1:30)
Open with the starting point (strip at the top): E0 improved KITTI D1 2.86 → 1.86 % but Middlebury BP-2 went 1.10 → 2.42 %.
Goal: keep the KITTI gain, forget less. Each method tests one explanation of the forgetting; bars = trainable share of E0.
- E1 BN locked (39.2 M, 100 %): batch-1 KITTI crops shift BN statistics and every later layer → is drift the cause?
- E2 GRU only (16.4 M, 42 %): network still sees and matches as before, only refinement learns → is KITTI only a
  refinement problem?
- E3a LoRA everywhere (1.86 M, 4.7 %, 217 layers, rank 8): low-rank updates can only nudge weights → does limiting the
  update limit forgetting?
- E3b LoRA on GRU (0.67 M, 1.7 %, 40 layers): smallest change that still adapts → how little capacity is enough?
Shared hypothesis: fewer, better-placed trainable parameters → less forgetting. Caveat: LoRA lr 1e-4 vs 1e-5.

### 15. What each run changes (~1:00)
Table: runs × network parts. DepthAnything frozen everywhere. E0 trains everything else with BN statistics updating; E1 is
the same with BN locked (isolates statistics drift). E2 freezes features + cost filtering, trains only GRU + upsampler.
E3a / E3b learn low-rank LoRA corrections instead of changing weights: E3a on every layer E1 trains (= low-rank E1),
E3b only on the layers E2 trains (= low-rank E2). Trainable parameters 39.2 M → 16.4 M → 1.86 M → 0.67 M.

### 16. How the methods work (~1:15)
- BN locking: BatchNorm normalises with running mean/variance from pretraining; batch 1 on KITTI crops (road, sky, cars)
  drags them toward KITTI and shifts every later layer. Locked = eval mode, weights still learn.
- Refinement only: features + cost volume decide what can be matched, the GRU how to refine. Bet: KITTI needs new
  refinement, not new matching.
- LoRA: y = W·x + (α/r)·B(A·x), W frozen, r = 8, α = 16 (scale 2). B starts at zero → step 0 is exactly the zero-shot model.
  After training B·A is merged into W → same size and speed as the original.
- Common bet: change less → forget less. Next slide tests it.

### 17. Trade-off chart (~1:45), the main slide
Two panels, both: x = KITTI-15 D1, y = Middlebury BP-2, bottom-left = best.
- Left (log scale, all methods): classical BM (23.2 / 29.0 %) and SGBM (18.1 / 23.6 %) far up-right; every learned model
  sits together near the origin (E1–E3b drawn as one dot).
- Right (linear zoom, learned only): zero-shot bottom-right (best Middlebury, weaker KITTI); E0 up-left; E1, E2, E3a, E3b in
  a tight cluster, slightly better than E0 on both but still ~2× zero-shot Middlebury error (E1 furthest left = best KITTI;
  E3b highest = worst Middlebury of the four). Even 0.67 M LoRA parameters don't protect Middlebury.
- Perspective: the fine-tuning cost is real, but even the worst fine-tuned model is ~10× better than classical stereo.
- E0 vs E1 is not a clean BN test (E1 also trained longer).

### 18. Per-scene (~1:15)
Sorted by how much worse E1 is than zero-shot. Biggest (×3–3.5): ArtL, PianoL, Piano, Shelves: thin rods, legs, cables,
fine edges. Large textured scenes (Vintage, Teddy, Playroom) barely change; Playroom slightly improves.
Hypothesis (unverified): KITTI’s sparse LiDAR labels (roads, cars) rarely cover thin structures → model learns to smooth
them. Would need error maps / a thin-structure mask.

---

## Wrap-up (slides 19–21, ~2.5 min)

### 19. Findings (~1:00)
1) Zero-shot results reproduce. 2) KITTI fine-tuning works on KITTI (35–48 % lower D1). 3) It costs ~2× Middlebury error,
whatever we train. 4) Damage shows up even in short runs (E0: 350 steps) and hits thin structures. 5) E1 best on KITTI, but gaps to LoRA ≤ 0.08 pts from one seed each → not a reliable winner.
Not claimed: that BN locking helps, or that protocol B beats A.

### 20. Phone demo (~1:00)
Two phone photos, moved sideways by hand. The model assumes rectified input (same row). Raw rows were ~9 px off (the
phone turned slightly). We rectified ourselves: SIFT matches → essential matrix → OpenCV stereoRectify; rows within
0.3 px median. Both outputs look plausible because the DepthAnything prior gives a believable shape even when matching is
off: robust, but a nice-looking map is not proof of correct geometry. Relative depth only (no baseline, no GT).

### 21. Thank you / Q&A
Message: specializing a foundation stereo model costs generality elsewhere; in our experiments even LoRA did not avoid it.

---

## Likely questions (short answers)

- **Is 1.10 vs 1.1 really a reproduction?** Yes within rounding; with per-scene averaging it’s 1.15. We say “within ~0.05 pts”.
- **Why not evaluate on the KITTI leaderboard?** Test labels are hidden and submissions are limited; we use the public
  training images with a held-out split, so our KITTI numbers are not leaderboard numbers.
- **Did you train FSD or the base model?** No. NVIDIA’s released checkpoint; we only fine-tune it.
- **Why does LoRA not help?** Our reading: even a small change to the refinement stage is enough to learn KITTI-specific
  behaviour (E3b, 0.67 M parameters on the GRU only, is the worst on Middlebury). Restricting capacity ≠ restricting what is learned.
- **Is BN locking useful?** Unclear: E1 vs E0 also differs in training length; needs a same-step comparison.
- **How significant are the differences?** Paired bootstrap over images (10k resamples): E1 vs others on KITTI was significant,
  but seed-to-seed noise isn’t covered with one seed. The ~2× Middlebury effect is robust (95 % CI of the ratio 1.6–2.5, far above 1).
- **Protocol A vs B?** Different test sets and validation sets → treated as comparable.
- **What would you try next?** Weight averaging between zero-shot and fine-tuned weights; mixing a little synthetic/Middlebury-like
  data during fine-tuning (rehearsal); more seeds.
- **Runtime?** Paper: 0.7 s for 375×1242 on an A100. Ours: 5.4 s for 960×540 on a GTX 1650 in fp32.
