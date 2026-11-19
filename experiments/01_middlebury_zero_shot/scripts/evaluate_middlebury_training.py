import argparse
import csv
import os
import sys

import imageio.v2 as imageio
import numpy as np


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().rstrip().decode("ascii")
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header in {path}: {header}")
        dims = f.readline().decode("ascii").strip().split()
        if len(dims) != 2:
            raise ValueError(f"Invalid PFM dimensions in {path}.")
        width, height = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if header == "PF" else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        if header == "PF":
            data = data[:, :, 0]
        return data.astype(np.float32)


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    valid = np.isfinite(pred) & np.isfinite(gt) & mask
    if valid.sum() == 0:
        return {
            "valid_pixels": 0,
            "epe": float("nan"),
            "rmse": float("nan"),
            "bp0.5": float("nan"),
            "bp1.0": float("nan"),
            "bp2.0": float("nan"),
            "bp4.0": float("nan"),
        }
    err = np.abs(pred[valid] - gt[valid])
    return {
        "valid_pixels": int(valid.sum()),
        "epe": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bp0.5": float((err > 0.5).mean() * 100.0),
        "bp1.0": float((err > 1.0).mean() * 100.0),
        "bp2.0": float((err > 2.0).mean() * 100.0),
        "bp4.0": float((err > 4.0).mean() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True, type=str)
    parser.add_argument("--gt_root", required=True, type=str)
    parser.add_argument("--algo_name", default="FS", type=str)
    parser.add_argument("--out_csv", required=True, type=str)
    args = parser.parse_args()

    pred_training_root = os.path.join(args.pred_root, "trainingH")
    gt_training_root = os.path.join(args.gt_root, "trainingH")
    if not os.path.isdir(pred_training_root):
        raise FileNotFoundError(f"Missing predictions: {pred_training_root}")
    if not os.path.isdir(gt_training_root):
        raise FileNotFoundError(f"Missing GT: {gt_training_root}")

    rows = []
    scene_names = sorted(
        [
            d
            for d in os.listdir(gt_training_root)
            if os.path.isdir(os.path.join(gt_training_root, d))
        ]
    )
    for scene in scene_names:
        pred_path = os.path.join(
            pred_training_root, scene, f"disp0{args.algo_name}.pfm"
        )
        gt_path = os.path.join(gt_training_root, scene, "disp0GT.pfm")
        nocc_mask_path = os.path.join(gt_training_root, scene, "mask0nocc.png")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"Missing prediction: {pred_path}")
        pred = read_pfm(pred_path)
        gt = read_pfm(gt_path)
        nocc_mask_img = imageio.imread(nocc_mask_path)
        nocc_mask = nocc_mask_img == 255

        all_mask = np.ones_like(nocc_mask, dtype=bool)
        m_nonocc = compute_metrics(pred, gt, nocc_mask)
        m_all = compute_metrics(pred, gt, all_mask)

        rows.append({"scene": scene, "mask": "nonocc", **m_nonocc})
        rows.append({"scene": scene, "mask": "all", **m_all})

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = [
        "scene",
        "mask",
        "valid_pixels",
        "epe",
        "rmse",
        "bp0.5",
        "bp1.0",
        "bp2.0",
        "bp4.0",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for mask_name in ["nonocc", "all"]:
        subset = [r for r in rows if r["mask"] == mask_name]
        if not subset:
            continue
        print(f"\nSummary ({mask_name}) over {len(subset)} scenes:")
        for key in ["epe", "rmse", "bp0.5", "bp1.0", "bp2.0", "bp4.0"]:
            vals = np.array([float(r[key]) for r in subset], dtype=np.float64)
            print(f"  {key}: {safe_mean(vals):.4f}")
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
