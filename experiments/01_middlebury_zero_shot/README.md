# 01 · Zero-shot reproduction on Middlebury

Reproduces the paper's zero-shot Middlebury number with the released FoundationStereo checkpoint.

## Run (Google Colab, GPU)
Upload two files and run all cells:
- `foundationstereo_middlebury_reproduction.ipynb`
- `../foundationstereo_source_bundle.zip` (FoundationStereo code: `core/`, `depth_anything/`, `dinov2/`, `Utils.py`)

The notebook downloads the official checkpoint (`23-51-11/model_best_bp2.pth`) and Middlebury Eval3 half-resolution data,
runs all 30 pairs (15 trainingH + 15 testH) and scores the 15 training pairs.

Settings: scale 1.0, 32 GRU iterations, mixed precision, non-occluded pixels (`mask0nocc == 255`).

## Result
| | BP-2 (pixel-weighted) | BP-2 (mean of scenes) | EPE |
|---|---|---|---|
| Ours | **1.0987 %** | 1.1475 % | 0.3634 px |
| Paper (Table 2) | 1.1 % | | |

## Files
- `results/middlebury_training_half_nonocc.csv` — per-scene EPE, BP-0.5/1/2/4, valid-pixel counts
- `results/inference_summary.csv` — per-pair run time and image size
- `scripts/` — the same pipeline as plain Python (`run_foundationstereo_middlebury.py` inference,
  `evaluate_middlebury_training.py` metrics, `kaggle_foundationstereo_middlebury.py` single-file Kaggle runner)
- Raw predictions (123 MB) are kept locally in `data/archives/` and not pushed.
