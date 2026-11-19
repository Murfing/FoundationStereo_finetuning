# FoundationStereo inference

`infer.py` runs any of the project's models on a stereo pair or on a folder of pairs. It saves the disparity maps and, when ground truth is given, prints EPE / D1 / BP-2.

## Colab version
Upload `foundationstereo_inference_colab.ipynb` and `../experiments/foundationstereo_source_bundle.zip` to Colab. Choose a GPU runtime, then edit **cell 1**; every line to change is marked `# <<<`. Then Run all.
- The zero-shot model downloads automatically.
- E1/E2/E3 are read from `MyDrive/foundationstereo_e1_e4_lora/checkpoints/`.
- E0_A/E0_B must be uploaded to `MyDrive/foundationstereo_models/`.
- The default input is a Middlebury demo with ground truth. Set `INPUT_MODE` to `"upload"`, `"pair"`, `"folder"` or `"scenes"` to use your own images.

## 1. Put the model files in `inference/models/` (local use)

| `--model` | File it looks for | Status |
|---|---|---|
| `zero_shot` (zs, base) | `models/zero_shot_foundationstereo.pth` + `models/cfg.yaml` | present |
| `E0_A` (e0) | `models/E0_A_full_ft_protocol_a.pth` | not available (never saved outside Colab) |
| `E0_B` | `models/E0_B_full_ft_protocol_b.pth` | present |
| `E1` | `models/E1_bn_locked.pth` | present (run 2) |
| `E2` | `models/E2_gru_only.pth` | present (run 2) |
| `E3a` | `models/E3a_lora_all.pth` (+ adapter fallback) | present (run 2) |
| `E3b` | `models/E3b_lora_gru.pth` (+ adapter fallback) | present (run 2) |

**The full command reference is in [RUN_COMMANDS.txt](RUN_COMMANDS.txt).**

If your files have other names or live somewhere else, edit the paths in `models.json`, or pass `--ckpt <file>` directly.
Run `python inference/infer.py --list` to check which models are found.

## 2. Commands (run from the `Computer vision` folder)

```
# one pair, one model
python inference/infer.py --model E1 --left path/left.png --right path/right.png

# same pair, several models side by side (writes a comparison grid)
python inference/infer.py --model zero_shot,E0_A,E1,E2,E3a,E3b --left L.png --right R.png

# every model that is found
python inference/infer.py --model all --left L.png --right R.png

# with ground truth (KITTI 16-bit png, or .pfm) -> EPE / D1 / BP-2 printed + metrics.csv
python inference/infer.py --model E1,E3a --left image_2/000010_10.png --right image_3/000010_10.png --gt disp_occ_0/000010_10.png

# a whole KITTI 2015 folder
python inference/infer.py --model E1 --left_dir KITTI2015/training/image_2 --right_dir KITTI2015/training/image_3 --gt_dir KITTI2015/training/disp_occ_0 --pattern "*_10.png"

# Middlebury trainingH (scene folders with im0.png/im1.png; GT folder with disp0GT.pfm + mask0nocc.png)
python inference/infer.py --model zero_shot,E1 --scenes_dir MiddEval3/trainingH --gt_root MiddEval3_GT/trainingH

# any checkpoint / any LoRA adapter not in the registry
python inference/infer.py --ckpt D:/somewhere/my_model.pth --left L.png --right R.png
python inference/infer.py --adapter inference/models/E3b_lora_gru_lora_adapter.pth --left L.png --right R.png
```

## 3. Useful options

| Option | Meaning |
|---|---|
| `--out_dir DIR` | where results go (default `inference/outputs`) |
| `--iters 32` | GRU iterations (32 = what all our evaluations used) |
| `--scale 0.5` | run at half resolution (less GPU memory). The disparity is upsampled back and rescaled. |
| `--hiera` | hierarchical inference for >1K images |
| `--device cpu` | run without a GPU (slow) |
| `--fp32` | turn off fp16. Pairs that give NaN in fp16 (e.g. on a GTX 1650) are rerun in fp32 automatically. |
| `--save_pfm`, `--save_png16` | also write .pfm / KITTI-format 16-bit png |
| `--focal F --baseline B` | also save depth in metres (`depth = F*B/disp`) |
| `--limit N` | only the first N pairs |
| `--no_vis` | skip the colour images |

## 4. Outputs

```
outputs/
  <model>/<pair>_disp.npy     float32 disparity (pixels, full input resolution)
  <model>/<pair>_disp.png     colour map (TURBO, range from GT if given)
  comparison/<pair>_compare.png   left | GT | each model (only when >1 model), with per-model metrics in the label
  metrics.csv                 one row per model x pair (EPE, D1, BP-1/2/3, time)
```

The metric definitions match the notebooks:
- D1: error > 3 px and > 5 % of the true disparity.
- BP-2: error > 2 px.
- With a mask, only pixels where the mask is 255 count (Middlebury `mask0nocc`).

The summary aggregates pixel-weighted, the same way as the notebooks.

## Notes

- The full checkpoints are ~1.5–3 GB. A 4 GB GPU can be tight at full KITTI resolution. If you run out of memory, use `--scale 0.5` or `--device cpu`.
- The models are loaded one at a time, so comparing several models doesn't need more memory.
- Full LoRA checkpoints already have the adapter merged in. The adapter path is only used when the full file is missing.
