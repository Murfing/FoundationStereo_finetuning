import argparse
import csv
import os
import sys
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


def write_pfm(path: str, image: np.ndarray, scale: float = 1.0) -> None:
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.ndim != 2:
        raise ValueError("PFM writer expects a 2D float32 array.")
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        endian_scale = -scale if image.dtype.byteorder == "<" or (
            image.dtype.byteorder == "=" and sys.byteorder == "little"
        ) else scale
        f.write(f"{endian_scale}\n".encode("ascii"))
        np.flipud(image).tofile(f)


def vis_disparity(disp: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disp)
    if valid.sum() == 0:
        return np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
    dmin = float(disp[valid].min())
    dmax = float(disp[valid].max())
    if dmax <= dmin:
        norm = np.zeros_like(disp, dtype=np.uint8)
    else:
        norm = ((disp - dmin) / (dmax - dmin) * 255.0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color[:, :, ::-1]


def build_model(repo_root: str, ckpt_path: str, valid_iters: int, mixed_precision: int):
    sys.path.insert(0, repo_root)
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = int(valid_iters)
    cfg["mixed_precision"] = bool(mixed_precision)

    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    return model, InputPadder


def run_pair(
    model,
    InputPadder,
    left_file: str,
    right_file: str,
    valid_iters: int,
    scale: float,
    hiera: int,
):
    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)
    h_full, w_full = img0.shape[:2]
    if scale <= 0 or scale > 1:
        raise ValueError("scale must be in (0, 1].")
    if scale != 1:
        img0 = cv2.resize(img0, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = img0.shape[:2]
    t0 = time.perf_counter()
    with torch.no_grad():
        x0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        x1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(x0.shape, divis_by=32, force_square=False)
        x0, x1 = padder.pad(x0, x1)
        with torch.cuda.amp.autocast(True):
            if hiera:
                disp = model.run_hierachical(
                    x0, x1, iters=valid_iters, test_mode=True, small_ratio=0.5
                )
            else:
                disp = model.forward(x0, x1, iters=valid_iters, test_mode=True)
        disp = padder.unpad(disp.float())
    elapsed = time.perf_counter() - t0
    disp_np = disp.detach().cpu().numpy().reshape(h, w).astype(np.float32)
    if scale != 1:
        disp_np = cv2.resize(
            disp_np,
            dsize=(w_full, h_full),
            interpolation=cv2.INTER_LINEAR,
        )
        disp_np = disp_np / float(scale)
    return disp_np, elapsed


def list_scenes(split_root: str):
    return sorted(
        [
            d
            for d in os.listdir(split_root)
            if os.path.isdir(os.path.join(split_root, d))
        ]
    )


def parse_csv_list(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--middlebury_data_root", required=True, type=str)
    parser.add_argument("--out_root", required=True, type=str)
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--algo_name", type=str, default="FS")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--mixed_precision", type=int, default=0)
    parser.add_argument("--splits", type=str, default="trainingH,testH")
    parser.add_argument("--scenes", type=str, default="")
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model, InputPadder = build_model(
        args.repo_root,
        args.ckpt_path,
        args.valid_iters,
        args.mixed_precision,
    )
    os.makedirs(args.out_root, exist_ok=True)
    summary_rows = []

    split_list = parse_csv_list(args.splits)
    scene_allow = set(parse_csv_list(args.scenes)) if args.scenes else None

    for split in split_list:
        split_in = os.path.join(args.middlebury_data_root, split)
        if not os.path.isdir(split_in):
            raise FileNotFoundError(f"Missing split directory: {split_in}")
        for scene in list_scenes(split_in):
            if scene_allow is not None and scene not in scene_allow:
                continue
            scene_in = os.path.join(split_in, scene)
            left_file = os.path.join(scene_in, "im0.png")
            right_file = os.path.join(scene_in, "im1.png")
            scene_out = os.path.join(args.out_root, split, scene)
            os.makedirs(scene_out, exist_ok=True)
            disp_path = os.path.join(scene_out, f"disp0{args.algo_name}.pfm")
            time_path = os.path.join(scene_out, f"time{args.algo_name}.txt")
            vis_path = os.path.join(scene_out, f"vis0{args.algo_name}.png")
            if args.overwrite == 0 and os.path.isfile(disp_path) and os.path.isfile(time_path):
                print(f"[skip] {split}/{scene} already exists")
                continue

            disp, elapsed = run_pair(
                model=model,
                InputPadder=InputPadder,
                left_file=left_file,
                right_file=right_file,
                valid_iters=args.valid_iters,
                scale=args.scale,
                hiera=args.hiera,
            )
            write_pfm(disp_path, disp)
            with open(time_path, "w", encoding="utf-8") as f:
                f.write(f"{elapsed:.6f}\n")
            imageio.imwrite(vis_path, vis_disparity(disp))

            summary_rows.append(
                {
                    "split": split,
                    "scene": scene,
                    "left_file": left_file,
                    "right_file": right_file,
                    "disp_path": disp_path,
                    "time_seconds": round(elapsed, 6),
                }
            )
            print(f"[done] {split}/{scene} in {elapsed:.3f}s")

    csv_path = os.path.join(args.out_root, "inference_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "scene",
                "left_file",
                "right_file",
                "disp_path",
                "time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
