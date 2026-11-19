# 02 · KITTI fine-tuning, Protocols A and B (E0)

Full fine-tuning of FoundationStereo on KITTI (everything except the frozen DepthAnything V2 backbone, 39.2 M parameters),
then re-evaluation on Middlebury to measure how much zero-shot generalization is lost.

## Run (Google Colab, GPU)
Upload `foundationstereo_kitti_finetuning.ipynb` and `../foundationstereo_source_bundle.zip`, then run all cells.
The notebook downloads KITTI Stereo 2012 + 2015 (training sets), the checkpoint and Middlebury, builds deterministic splits
(`np.random.default_rng(2025)`), fine-tunes, evaluates and exports checkpoints and a JSON summary.

| | Train | Validation (checkpoint selection) | Test |
|---|---|---|---|
| Protocol A | KITTI 2012, 194 pairs | KITTI 2015, 40 pairs | KITTI 2015, other 160 pairs |
| Protocol B | KITTI 2012, 155 pairs | KITTI 2012, 39 pairs | KITTI 2015, all 200 pairs |

Recipe: AdamW (lr 1e-5, wd 1e-5), batch 1, random 256×512 crops, smooth-L1 sequence loss (γ 0.9, 12 GRU iterations),
grad clip 1.0, AMP, validation every 50 steps, ReduceLROnPlateau (×0.5, patience 2), early stopping (patience 3).

## Results (all %, lower is better)
| | KITTI-15 test D1-all | Middlebury BP-2 |
|---|---|---|
| Zero-shot (A / B test set) | 2.8561 / 2.8307 | 1.0987 |
| E0 Protocol A | **1.8623** | 2.4225 |
| E0 Protocol B | **1.7005** | 2.4036 |

KITTI improves 35–40 %, Middlebury error more than doubles. A and B use different test sets → treated as comparable.

## Files
- `results/protocol_a_summary.json` — Protocol A metrics, validation curve, settings
- `results/protocol_b_summary.json` — Protocol B metrics with per-image rows
- `results/final_results_E0_A_B.json` — combined A + B summary
