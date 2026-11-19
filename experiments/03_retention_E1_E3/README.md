# 03 · Keeping zero-shot generalization: E1, E2, E3a, E3b (Protocol A)

Four lighter ways to fine-tune on KITTI, each testing one explanation of why E0 forgets Middlebury.
Same Colab setup, downloads, Protocol A split, loss, optimizer, scheduler and early stopping as
`../02_kitti_finetuning_E0/`; BatchNorm statistics locked in every run.

| Run | What is trained | Trainable params | LR |
|---|---|---|---|
| E1 · BN locked | everything except DepthAnything, BatchNorm statistics frozen | 39.2 M | 1e-5 |
| E2 · GRU only | GRU update block + upsampler (`update_block`, `spx_2_gru`, `spx_gru`, `stem_2`) | 16.4 M | 1e-5 |
| E3a · LoRA all | LoRA r = 8, α = 16 on 217 conv/linear layers outside DepthAnything | 1.86 M | 1e-4 |
| E3b · LoRA GRU | LoRA r = 8, α = 16 on the 40 GRU/upsampler layers | 0.67 M | 1e-4 |

## Run (Google Colab, GPU, Google Drive)
Upload `foundationstereo_e1_e3_lora.ipynb` and `../foundationstereo_source_bundle.zip` → GPU runtime → Run all → allow Drive.
Finished runs are skipped if Colab disconnects. Checkpoints (≈1.5 GB each, LoRA merged in) and LoRA adapters are written to
Google Drive.

## Results (run 1; KITTI-15 test 160 images, Middlebury 15 scenes; %, lower is better)
| Model | KITTI D1-all | Middlebury BP-2 |
|---|---|---|
| Zero-shot | 2.8561 | 1.0987 |
| E0 full fine-tune | 1.8623 | 2.4225 |
| E1 BN locked | **1.4979** | **2.1074** |
| E2 GRU only | 1.6191 | 2.2676 |
| E3a LoRA all | 1.5747 | 2.1765 |
| E3b LoRA GRU | 1.5752 | 2.3446 |

Every method keeps ~1.9–2.1× the zero-shot Middlebury error: restricting what is trained does not prevent forgetting.

## Files
- `results/run1/` — first run (numbers above): `e1_e4_lora_results.json` (metrics, curves, settings), `<run>_rows.json`
  (per image / per scene), `comparison_table.csv`
- `results/run2/` — second execution with the same seed (checkpoints in `inference/models/` come from this run)

**Naming:** experiments were renamed after the runs (E4 → E2, E5a → E3a, E5b → E3b). Files written by the notebook
keep the original names (`E4_gru_only_rows.json`, `E5a_*`, `E5b_*`, `e1_e4_lora_results.json`).
