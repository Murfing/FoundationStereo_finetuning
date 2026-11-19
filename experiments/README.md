# Experiments

All training and evaluation ran on Google Colab GPUs. Each folder has the notebook, a README and its result files.

| Folder | What | Key result |
|---|---|---|
| `01_middlebury_zero_shot/` | Reproduce the paper's zero-shot Middlebury number | BP-2 1.0987 % (paper 1.1 %) |
| `02_kitti_finetuning_E0/` | Full fine-tuning on KITTI, Protocols A and B | KITTI D1 2.86 → 1.86 %, Middlebury BP-2 1.10 → 2.42 % |
| `03_retention_E1_E3/` | BN locking, GRU-only, LoRA (E1, E2, E3a, E3b) | All keep ~2× Middlebury error |

`foundationstereo_source_bundle.zip` is the FoundationStereo code packed for Colab (`core/`, `depth_anything/`, `dinov2/`,
`Utils.py`, identical to `../FoundationStereo/`). Upload it next to whichever notebook you run.

The classical baselines (BM, SGBM) and all local inference are in `../inference/`.
