# FoundationStereo: zero-shot reproduction and KITTI fine-tuning

Course project (Computer Vision, instructor Prof. Sukhendu Das) by **Archit Shukla (CH23B072)** and **Akshit Singh (CH23B071)**.

We reproduce the zero-shot results of **FoundationStereo** (Wen et al., CVPR 2025, [arXiv:2501.09898](https://arxiv.org/abs/2501.09898),
[official code](https://github.com/NVlabs/FoundationStereo)), fine-tune it on KITTI, measure how much zero-shot generalization
is lost on Middlebury, and test four ways to reduce that loss (BatchNorm locking, refinement-only training, LoRA).
Classical stereo (OpenCV BM / SGBM) is evaluated on the same data as a reference.

## Main results
KITTI 2015 test = Protocol A held-out split (160 images); Middlebury = Eval3 trainingH, 15 scenes, half resolution,
non-occluded pixels. All numbers are errors in %, lower is better.

| Model | Trainable params | KITTI-15 D1-all | Middlebury BP-2 |
|---|---|---|---|
| BM (OpenCV, classical) | – | 23.19 | 29.03 |
| SGBM (OpenCV, classical) | – | 18.05 | 23.56 |
| FoundationStereo zero-shot (paper: 2.8 / 1.1) | – | 2.86 | **1.10** |
| E0 full fine-tune | 39.2 M | 1.86 | 2.42 |
| E1 BatchNorm locked | 39.2 M | **1.50** | 2.11 |
| E2 GRU + upsampler only | 16.4 M | 1.62 | 2.27 |
| E3a LoRA (r = 8) on all layers | 1.86 M | 1.57 | 2.18 |
| E3b LoRA (r = 8) on GRU only | 0.67 M | 1.58 | 2.34 |

Findings: the paper's zero-shot numbers reproduce; KITTI fine-tuning lowers KITTI D1 by 35–48 % but roughly doubles the
Middlebury error, and none of the lighter adaptation methods prevents this; every learned model is ~10× better than
classical stereo. Full analysis, caveats and statistics: [`docs/my_notes.md`](docs/my_notes.md).

## Pipeline
```
 rectified stereo pair ──► FoundationStereo (frozen DepthAnything V2 + trainable CNN features
                           → hybrid cost volume → attentive cost filtering → ConvGRU refinement) ──► disparity
 1. Zero-shot:   released checkpoint → Middlebury + KITTI evaluation          (experiments/01)
 2. Fine-tune:   KITTI 2012 train → select on KITTI-15 val → test KITTI-15     (experiments/02, E0, Protocols A/B)
 3. Retention:   same, with BN locking / GRU-only / LoRA → test KITTI + Middlebury (experiments/03, E1–E3b)
 4. Baselines:   OpenCV BM / SGBM on the same test sets                        (inference/classical_baseline.py)
 5. Inference:   any model on any rectified pair; phone-photo rectification    (inference/)
```

## Repository layout
```
├── FoundationStereo/        official NVlabs code (commit 6e88068) + one local patch (LOCAL_CHANGES_vs_upstream_6e88068.patch)
├── experiments/
│   ├── 01_middlebury_zero_shot/   Colab notebook, scripts, per-scene metrics
│   ├── 02_kitti_finetuning_E0/    Colab notebook, Protocol A / B results
│   ├── 03_retention_E1_E3/        Colab notebook, results of E1, E2, E3a, E3b (two runs)
│   └── foundationstereo_source_bundle.zip   code bundle uploaded next to each notebook
├── inference/
│   ├── infer.py                   run any model (zero-shot / fine-tuned / LoRA) on images, with metrics if GT is given
│   ├── classical_baseline.py      OpenCV BM / SGBM evaluated with the same metrics and splits
│   ├── check_rectification.py, rectify_pair.py   check / rectify phone-photo pairs
│   ├── foundationstereo_inference_colab.ipynb    inference on Colab
│   ├── models.json                model registry (checkpoint files are not in git)
│   ├── outputs/classical_baseline/   BM / SGBM summaries, per-image metrics, Middlebury visualizations
│   ├── test_images/               demo pairs (Middlebury Adirondack, official demo)
│   └── RUN_COMMANDS.txt           every command with explanations
├── data/README.md           dataset download links (data itself not in git)
├── docs/                    lab notes (all experiments + critical analysis), presentation speaker notes
└── requirements.txt
```

## How to run
**Training / evaluation (Google Colab, GPU):** open the notebook in `experiments/0x_*/`, upload it together with
`experiments/foundationstereo_source_bundle.zip`, run all cells. Notebooks download data and the checkpoint themselves.

**Local inference:**
```bash
pip install -r requirements.txt
# put checkpoints in inference/models/ (see "Models"), then
python inference/infer.py --list
python inference/infer.py --model zero_shot --left L.png --right R.png
python inference/infer.py --model zero_shot,E1 --test_images            # every pair in inference/test_images/
python inference/classical_baseline.py --datasets middlebury,kitti --methods sgbm,bm --download_kitti
```
GPUs with ≤ 4 GB may need `--fp32 --scale 0.5` (lower accuracy). See `inference/RUN_COMMANDS.txt`.

## Models
Checkpoints are 1.5–3.3 GB and are not stored in git.
- **Zero-shot (NVIDIA):** `23-51-11` ViT-L from the [official release](https://github.com/NVlabs/FoundationStereo#model-weights)
  → save as `inference/models/zero_shot_foundationstereo.pth` with its `cfg.yaml`.
- **Our fine-tuned models** (E0_B, E1, E2, E3a, E3b; LoRA adapters 3–8 MB): available from the authors on request
  (Kaggle link to be added). `models.json` lists the expected file names.

## Data
Middlebury v3 and KITTI 2012/2015, download links in [`data/README.md`](data/README.md).

## Acknowledgements and license
- FoundationStereo code and weights © NVIDIA, [NVIDIA license](FoundationStereo/LICENSE): non-commercial research use only;
  our fine-tuned weights are derivatives and fall under the same terms.
- DepthAnything V2 and DINOv2 are included through the FoundationStereo code with their own licenses.
- Datasets: Middlebury Stereo (Scharstein et al.), KITTI (Geiger et al., Menze & Geiger).
- AI assistance: [Claude Code](https://claude.com/claude-code) (Anthropic) was used for code, analysis and documentation support.
