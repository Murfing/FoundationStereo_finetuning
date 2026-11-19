# Data (not included in the repository)

Datasets are downloaded by the Colab notebooks automatically. For local use, place them here:

```
data/
├── middlebury/raw/data-H/MiddEval3/trainingH/<scene>/{im0.png, im1.png, calib.txt}
├── middlebury/raw/gt0-H/MiddEval3/trainingH/<scene>/{disp0GT.pfm, mask0nocc.png}
└── kitti/raw/kitti2015/training/{image_2, image_3, disp_occ_0, disp_noc_0}/
```

| Dataset | Used for | Download |
|---|---|---|
| Middlebury Stereo Evaluation v3, half resolution (images) | zero-shot / retention evaluation (15 trainingH scenes) | https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip |
| Middlebury v3 ground truth, half resolution | same | https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip |
| KITTI Stereo 2015 (`data_scene_flow.zip`, ~1.7 GB) | validation (40) and test (160), Protocols A/B | https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip |
| KITTI Stereo 2012 (`data_stereo_flow.zip`) | fine-tuning (194 pairs) | https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip |

Dataset pages: [Middlebury v3](https://vision.middlebury.edu/stereo/submit3/) ·
[KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) ·
[KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo). Please follow each dataset's terms of use.

`inference/classical_baseline.py --download_kitti` downloads KITTI 2015 into `data/kitti/raw/` and extracts only the folders it needs.
