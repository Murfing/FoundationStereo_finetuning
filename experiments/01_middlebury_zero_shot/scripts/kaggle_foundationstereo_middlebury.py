"""
Kaggle runner for the FoundationStereo Middlebury reproduction.

Run in a Kaggle notebook with Internet enabled and a GPU accelerator:

    !python /kaggle/working/kaggle_foundationstereo_middlebury.py

The script uses the paper-style half-resolution setup:
  - Middlebury Eval3 half-resolution input
  - scale=1.0 (no additional downsampling)
  - 32 refinement iterations
  - non-occluded training metrics

The 15 test scenes have hidden ground truth. Predictions are still generated
for all 30 scenes and can be packaged for later submission.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


PROJECT = Path("/kaggle/working/foundationstereo_middlebury")
REPO = PROJECT / "FoundationStereo"
DATA = PROJECT / "data" / "middlebury"
CHECKPOINT = PROJECT / "checkpoints" / "23-51-11"
OUTPUT = PROJECT / "outputs" / "middlebury_half"

DATA_URL = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip"
GT_URL = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip"
CFG_ID = "1tidGICH1_kTUUqi42aboKscuMY4IK_Xr"
MODEL_ID = "1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def install_dependencies() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "gdown",
            "omegaconf",
            "imgaug",
            "ruamel.yaml",
            "trimesh",
            "imageio",
            "open3d",
            "transformations",
            "einops",
        ]
    )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already downloaded: {destination}", flush=True)
        return
    print(f"Downloading: {url}", flush=True)
    urllib.request.urlretrieve(url, destination)


def prepare_repository() -> None:
    if not (REPO / ".git").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/FoundationStereo.git", str(REPO)])


def prepare_data() -> tuple[Path, Path]:
    data_zip = DATA / "MiddEval3-data-H.zip"
    gt_zip = DATA / "MiddEval3-GT0-H.zip"
    data_extract = DATA / "data-H"
    gt_extract = DATA / "gt0-H"

    download(DATA_URL, data_zip)
    download(GT_URL, gt_zip)

    if not (data_extract / "MiddEval3" / "trainingH").exists():
        with zipfile.ZipFile(data_zip) as archive:
            archive.extractall(data_extract)
    if not (gt_extract / "MiddEval3" / "trainingH").exists():
        with zipfile.ZipFile(gt_zip) as archive:
            archive.extractall(gt_extract)

    return data_extract / "MiddEval3", gt_extract / "MiddEval3"


def prepare_checkpoint() -> Path:
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    cfg = CHECKPOINT / "cfg.yaml"
    model = CHECKPOINT / "model_best_bp2.pth"
    if not cfg.exists():
        run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={CFG_ID}", "-O", str(cfg)])
    if not model.exists():
        run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={MODEL_ID}", "-O", str(model)])
    return model


def run_inference(data_root: Path, checkpoint: Path, output_root: Path, valid_iters: int) -> None:
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(REPO))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    output_root.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(checkpoint.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["mixed_precision"] = True
    model = FoundationStereo(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.cuda().eval()

    def write_pfm(path: Path, image: np.ndarray) -> None:
        image = np.asarray(image, dtype=np.float32)
        with path.open("wb") as f:
            f.write(b"Pf\n")
            f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
            f.write(b"-1.0\n")
            np.flipud(image).tofile(f)

    def run_pair(left_path: Path, right_path: Path):
        left = imageio.imread(left_path)
        right = imageio.imread(right_path)
        height, width = left.shape[:2]
        left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        start = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            disparity = model.forward(
                left_tensor,
                right_tensor,
                iters=valid_iters,
                test_mode=True,
            )
        elapsed = time.perf_counter() - start
        disparity = padder.unpad(disparity.float())
        disparity = disparity.detach().cpu().numpy().reshape(height, width).astype(np.float32)
        return disparity, elapsed

    rows = []
    for split in ["trainingH", "testH"]:
        for scene_dir in sorted((data_root / split).iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_out = output_root / split / scene_dir.name
            scene_out.mkdir(parents=True, exist_ok=True)
            prediction_path = scene_out / "disp0FS-paper.pfm"
            time_path = scene_out / "timeFS-paper.txt"
            if prediction_path.exists() and time_path.exists():
                print(f"[skip] {split}/{scene_dir.name}", flush=True)
                continue
            disparity, elapsed = run_pair(scene_dir / "im0.png", scene_dir / "im1.png")
            write_pfm(prediction_path, disparity)
            time_path.write_text(f"{elapsed:.6f}\n", encoding="ascii")
            rows.append(
                {
                    "split": split,
                    "scene": scene_dir.name,
                    "seconds": elapsed,
                    "prediction": str(prediction_path),
                }
            )
            print(f"[done] {split}/{scene_dir.name}: {elapsed:.2f}s", flush=True)

    with (output_root / "inference_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "scene", "seconds", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def read_pfm(path: Path):
    import numpy as np

    with path.open("rb") as f:
        header = f.readline().rstrip().decode("ascii")
        if header != "Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = map(int, f.readline().decode("ascii").split())
        scale = float(f.readline().decode("ascii"))
        endian = "<" if scale < 0 else ">"
        values = np.fromfile(f, endian + "f")
    return np.flipud(values.reshape(height, width)).astype(np.float32)


def evaluate_training(output_root: Path, gt_root: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np

    rows = []
    for scene_dir in sorted((gt_root / "trainingH").iterdir()):
        if not scene_dir.is_dir():
            continue
        prediction = read_pfm(output_root / "trainingH" / scene_dir.name / "disp0FS-paper.pfm")
        ground_truth = read_pfm(scene_dir / "disp0GT.pfm")
        nonocc = imageio.imread(scene_dir / "mask0nocc.png") == 255
        valid = np.isfinite(prediction) & np.isfinite(ground_truth) & nonocc
        error = np.abs(prediction[valid] - ground_truth[valid])
        rows.append(
            {
                "scene": scene_dir.name,
                "valid_pixels": int(valid.sum()),
                "epe": float(error.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bp0.5": float(np.mean(error > 0.5) * 100),
                "bp1.0": float(np.mean(error > 1.0) * 100),
                "bp2.0": float(np.mean(error > 2.0) * 100),
                "bp4.0": float(np.mean(error > 4.0) * 100),
            }
        )

    metrics_dir = OUTPUT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "middlebury_training_half_nonocc.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPaper-style local result: Middlebury half-resolution, non-occ", flush=True)
    for key in ["epe", "rmse", "bp0.5", "bp1.0", "bp2.0", "bp4.0"]:
        print(f"{key}: {np.mean([row[key] for row in rows]):.4f}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        print("Warning: KAGGLE_KERNEL_RUN_TYPE is not set; continuing anyway.", flush=True)
    if not args.skip_install:
        install_dependencies()
    prepare_repository()
    data_root, gt_root = prepare_data()
    checkpoint = prepare_checkpoint()
    run_inference(data_root, checkpoint, OUTPUT / "predictions", args.valid_iters)
    evaluate_training(OUTPUT / "predictions", gt_root)
    print(f"\nAll outputs are under: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
