"""
FoundationStereo inference for the zero-shot model and every fine-tuned model from this project.

Pick the model(s) by name from models.json, give either one stereo pair or a folder of pairs,
and optionally ground truth to get EPE / D1 / BP-x metrics.

Examples (run from the "Computer vision" folder):

  # one pair, one model
  python inference/infer.py --model E1 --left L.png --right R.png

  # compare several models on the same pair (comma list, or "all")
  python inference/infer.py --model zero_shot,E1,E3a --left L.png --right R.png --gt disp.png

  # a KITTI 2015 folder, with metrics against disp_occ_0
  python inference/infer.py --model E1 --left_dir KITTI/training/image_2 --right_dir KITTI/training/image_3 ^
         --gt_dir KITTI/training/disp_occ_0 --pattern "*_10.png"

  # Middlebury-style folder (one sub-folder per scene with im0.png / im1.png [/ disp0GT.pfm, mask0nocc.png])
  python inference/infer.py --model all --scenes_dir MiddEval3/trainingH --gt_root MiddEval3_GT/trainingH

  python inference/infer.py --list          # show the models and whether their files are found
"""

import argparse
import csv
import fnmatch
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent / "FoundationStereo"
DEFAULT_REGISTRY = HERE / "models.json"


# ----------------------------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------------------------
def load_registry(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = data["models"]
    aliases = {}
    for name, entry in models.items():
        aliases[name.lower()] = name
        for alias in entry.get("aliases", []):
            aliases[alias.lower()] = name
    return data, models, aliases


def resolve_path(text, base):
    if not text:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(text)))
    return p if p.is_absolute() else (base / p).resolve()


def first_existing(value, registry_dir):
    """'path'/'adapter' may be one path or a list of candidate paths; return (found or None, first candidate)."""
    candidates = [value] if isinstance(value, str) else list(value or [])
    paths = [resolve_path(c, registry_dir) for c in candidates if c]
    found = next((p for p in paths if p.is_file()), None)
    return found, (paths[0] if paths else None)


def model_files(entry, registry_dir):
    """Return (full checkpoint or None, adapter or None, expected full path, expected adapter path)."""
    full, full_expected = first_existing(entry.get("path"), registry_dir)
    adapter, adapter_expected = first_existing(entry.get("adapter"), registry_dir)
    return full, adapter, full_expected, adapter_expected


def print_model_list(models, registry_dir, base_ckpt):
    print(f"Registry: {registry_dir / 'models.json'}")
    print(f"Base (zero-shot) checkpoint: {base_ckpt}  [{'found' if base_ckpt.is_file() else 'MISSING'}]\n")
    print(f"{'name':<11} {'aliases':<28} {'status':<40} description")
    print("-" * 120)
    for name, entry in models.items():
        full, adapter, full_expected, _ = model_files(entry, registry_dir)
        if full:
            status = f"full checkpoint ({full.stat().st_size / 1024**3:.2f} GiB)"
        elif adapter and base_ckpt.is_file():
            status = "LoRA adapter + base checkpoint"
        else:
            status = f"MISSING ({full_expected.name if full_expected else '?'})"
        print(f"{name:<11} {', '.join(entry.get('aliases', [])):<28} {status:<40} {entry.get('description', '')}")


# ----------------------------------------------------------------------------------------------
# FoundationStereo import (stubs for optional packages that Utils.py imports but inference never uses)
# ----------------------------------------------------------------------------------------------
def import_foundation_stereo(repo_root):
    for optional in ["open3d", "trimesh", "joblib"]:
        try:
            __import__(optional)
        except ImportError:
            print(f"[info] '{optional}' not installed - using an empty stub (not needed for inference)")
            sys.modules[optional] = types.ModuleType(optional)
    repo_root = str(Path(repo_root).resolve())
    if not Path(repo_root, "core", "foundation_stereo.py").is_file():
        raise FileNotFoundError(f"FoundationStereo source not found at {repo_root} (use --repo_root)")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    return OmegaConf, FoundationStereo, InputPadder


# ----------------------------------------------------------------------------------------------
# LoRA (identical to the training notebook) - only used when a model is given as base + adapter
# ----------------------------------------------------------------------------------------------
class LoRAConv(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        conv = type(base)
        self.base = base
        self.lora_A = conv(base.in_channels, rank, base.kernel_size, stride=base.stride, padding=base.padding,
                           dilation=base.dilation, bias=False, padding_mode=base.padding_mode)
        self.lora_B = conv(rank, base.out_channels, 1, bias=False)
        self.scale = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @torch.no_grad()
    def merged(self):
        delta = self.lora_B.weight.flatten(1) @ self.lora_A.weight.flatten(1)
        self.base.weight += self.scale * delta.view_as(self.base.weight).to(self.base.weight.dtype)
        return self.base


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @torch.no_grad()
    def merged(self):
        self.base.weight += self.scale * (self.lora_B.weight @ self.lora_A.weight).to(self.base.weight.dtype)
        return self.base


def inject_lora(model, scope, rank, alpha):
    targets = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if full.startswith("feature.dino"):
                continue
            if scope is not None and full.split(".")[0] not in scope:
                continue
            if type(child) in (nn.Conv2d, nn.Conv3d) and child.groups == 1:
                targets.append((parent, child_name, LoRAConv))
            elif type(child) is nn.Linear:
                targets.append((parent, child_name, LoRALinear))
    for parent, child_name, wrapper in targets:
        setattr(parent, child_name, wrapper(getattr(parent, child_name), rank, alpha))
    return len(targets)


def apply_lora_adapter(model, adapter_path):
    blob = torch.load(adapter_path, map_location="cpu", weights_only=False)
    rank, alpha, scope, weights = blob["rank"], blob["alpha"], blob.get("scope"), blob["lora"]
    wrapped = inject_lora(model, scope, rank, alpha)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing_lora = [k for k in missing if ".lora_A." in k or ".lora_B." in k]
    if unexpected or missing_lora:
        raise RuntimeError(f"LoRA adapter does not match the model: {len(missing_lora)} missing, "
                           f"{len(unexpected)} unexpected keys (e.g. {(missing_lora + unexpected)[:3]})")
    merged = 0
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, (LoRAConv, LoRALinear)):
                setattr(parent, name, child.merged())
                merged += 1
    print(f"  LoRA adapter: rank {rank}, alpha {alpha}, scope {scope or 'all'}, "
          f"{wrapped} layers wrapped, {len(weights)} tensors loaded, {merged} merged")


# ----------------------------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------------------------
def build_model(fs, cfg_path, ckpt_path, adapter_path, device, fp16, iters):
    OmegaConf, FoundationStereo, _ = fs
    cfg = OmegaConf.load(cfg_path)
    cfg["vit_size"] = cfg.get("vit_size", "vitl")
    cfg["mixed_precision"] = bool(fp16)
    cfg["valid_iters"] = int(iters)
    model = FoundationStereo(cfg)

    weights_path = ckpt_path
    t0 = time.perf_counter()
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    info = {k: v for k, v in state.items() if k in ("experiment", "description", "best_step", "steps",
                                                       "global_step", "val_d1_noc", "best_val_d1_noc")} \
        if isinstance(state, dict) else {}
    del state, state_dict
    gc.collect()
    print(f"  weights: {weights_path} ({time.perf_counter() - t0:.1f}s)")
    if info:
        print("  checkpoint info: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    if adapter_path:
        apply_lora_adapter(model, adapter_path)
    model.to(device).eval()
    return model


# ----------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------
def read_rgb(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype == np.uint16:
        img = (img / 257.0).round().astype(np.uint8)
    return np.ascontiguousarray(img)


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().strip()
        if header not in (b"Pf", b"PF"):
            raise ValueError(f"Not a PFM file: {path}")
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        values = np.fromfile(f, ("<" if scale < 0 else ">") + "f4")
    channels = 3 if header == b"PF" else 1
    data = np.flipud(values.reshape(height, width, channels) if channels == 3 else values.reshape(height, width))
    return np.ascontiguousarray(data[..., 0] if channels == 3 else data).astype(np.float32)


def write_pfm(path, disp):
    disp = np.ascontiguousarray(disp.astype(np.float32))
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{disp.shape[1]} {disp.shape[0]}\n".encode())
        f.write(b"-1.0\n")
        np.flipud(disp).astype("<f4").tofile(f)


def read_gt(path):
    """KITTI 16-bit PNG (value / 256, 0 = invalid), PFM (inf = invalid) or .npy. Invalid -> NaN."""
    path = Path(path)
    if path.suffix.lower() == ".pfm":
        gt = read_pfm(path)
    elif path.suffix.lower() == ".npy":
        gt = np.load(path).astype(np.float32)
    else:
        raw = imageio.imread(path)
        if raw.ndim == 3:
            raw = raw[..., 0]
        gt = raw.astype(np.float32) / (256.0 if raw.dtype == np.uint16 else 1.0)
    gt = gt.astype(np.float32)
    gt[~np.isfinite(gt) | (gt <= 0)] = np.nan
    return gt


def colorize(disp, vmin, vmax):
    valid = np.isfinite(disp)
    norm = np.zeros(disp.shape, np.uint8)
    if vmax > vmin:
        norm = ((np.nan_to_num(disp, nan=vmin) - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    color[~valid] = 0
    return color


def disp_range(arrays):
    values = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a is not None and np.isfinite(a).any()]
                            or [np.zeros(1, np.float32)])
    return float(np.percentile(values, 1)), float(np.percentile(values, 99))


def label(img, text):
    img = img.copy()
    scale = max(0.5, img.shape[1] / 1600)
    thick = max(1, int(round(scale * 2)))
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (w + 12, h + 14), (0, 0, 0), -1)
    cv2.putText(img, text, (6, h + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------------------------
# Inference + metrics
# ----------------------------------------------------------------------------------------------
def predict(model, InputPadder, left, right, device, fp16, iters, scale, hiera):
    h_full, w_full = left.shape[:2]
    if scale != 1.0:
        left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = left.shape[:2]
    x0 = torch.from_numpy(left).to(device).float().permute(2, 0, 1)[None]
    x1 = torch.from_numpy(right).to(device).float().permute(2, 0, 1)[None]
    use_amp = fp16 and device.type == "cuda"
    model.args.mixed_precision = use_amp
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        if hiera:
            disp = model.run_hierachical(x0, x1, iters=iters, test_mode=True, small_ratio=0.5)
        else:
            padder = InputPadder(x0.shape, divis_by=32, force_square=False)
            p0, p1 = padder.pad(x0, x1)
            disp = padder.unpad(model.forward(p0, p1, iters=iters, test_mode=True).float())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    disp = disp.float().cpu().numpy().reshape(h, w)
    if scale != 1.0:
        disp = cv2.resize(disp, (w_full, h_full), interpolation=cv2.INTER_LINEAR) / scale
    return disp.astype(np.float32), elapsed


def compute_metrics(pred, gt, mask=None):
    valid = np.isfinite(gt) & np.isfinite(pred)
    if mask is not None:
        valid &= mask
    n = int(valid.sum())
    if n == 0:
        return None
    err = np.abs(pred[valid] - gt[valid])
    d1 = (err > 3.0) & (err / np.maximum(gt[valid], 1e-6) > 0.05)
    return {"valid_px": n, "epe": float(err.mean()), "d1": float(d1.mean() * 100),
            "bp1": float((err > 1).mean() * 100), "bp2": float((err > 2).mean() * 100),
            "bp3": float((err > 3).mean() * 100), "d1_bad_px": int(d1.sum()), "bp2_bad_px": int((err > 2).sum())}


# ----------------------------------------------------------------------------------------------
# Input pairs
# ----------------------------------------------------------------------------------------------
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff")
LEFT_NAMES = ["left", "im0", "l", "left_image", "img_left", "view0"]
RIGHT_NAMES = ["right", "im1", "r", "right_image", "img_right", "view1"]
GT_NAMES = ["gt", "disp_gt", "gt_disp", "disp0gt", "disparity", "disp_occ"]
MASK_NAMES = ["mask0nocc", "mask"]


def find_by_stem(folder, stems, exts):
    files = {f.stem.lower(): f for f in sorted(folder.iterdir()) if f.is_file() and f.suffix.lower() in exts}
    return next((files[s] for s in stems if s in files), None)


def read_calibration(folder):
    """K.txt as in FoundationStereo/assets: line 1 = 3x3 K (9 numbers), line 2 = baseline in metres."""
    k_file = folder / "K.txt"
    if not k_file.is_file():
        return None
    try:
        lines = [line for line in k_file.read_text().splitlines() if line.strip()]
        K = np.array(lines[0].split(), dtype=np.float64).reshape(3, 3)
        return float(K[0, 0]), float(lines[1].split()[0])
    except Exception as error:
        print(f"[warn] could not read {k_file} ({error}) - no depth for this pair")
        return None


def collect_pairs(args):
    pairs = []  # dicts: name, left, right, gt, mask, save_dir, calib
    if args.test_images:
        root = Path(args.test_images)
        if not root.is_dir():
            sys.exit(f"ERROR: test images folder not found: {root}")
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            left = find_by_stem(folder, LEFT_NAMES, IMAGE_EXT)
            right = find_by_stem(folder, RIGHT_NAMES, IMAGE_EXT)
            if left is None or right is None:
                print(f"[warn] {folder.name}: needs left.png + right.png (or im0.png + im1.png) - skipped")
                continue
            gt = find_by_stem(folder, GT_NAMES, IMAGE_EXT + (".pfm", ".npy"))
            mask = find_by_stem(folder, MASK_NAMES, IMAGE_EXT)
            pairs.append({"name": folder.name, "left": left, "right": right, "gt": gt, "mask": mask,
                          "save_dir": folder, "calib": read_calibration(folder)})
    elif args.left or args.right:
        if not (args.left and args.right):
            sys.exit("ERROR: --left and --right must be given together")
        pairs.append({"name": Path(args.left).stem, "left": Path(args.left), "right": Path(args.right),
                      "gt": Path(args.gt) if args.gt else None, "mask": Path(args.mask) if args.mask else None})
    elif args.left_dir or args.right_dir:
        if not (args.left_dir and args.right_dir):
            sys.exit("ERROR: --left_dir and --right_dir must be given together")
        left_dir, right_dir = Path(args.left_dir), Path(args.right_dir)
        names = sorted(f for f in os.listdir(left_dir)
                       if fnmatch.fnmatch(f, args.pattern) and f.lower().endswith((".png", ".jpg", ".jpeg", ".ppm", ".bmp")))
        for f in names:
            if not (right_dir / f).is_file():
                print(f"[warn] no right image for {f}, skipped")
                continue
            gt = Path(args.gt_dir) / f if args.gt_dir else None
            if gt is not None and not gt.is_file():
                print(f"[warn] no ground truth for {f} in {args.gt_dir} - prediction only")
                gt = None
            pairs.append({"name": Path(f).stem, "left": left_dir / f, "right": right_dir / f, "gt": gt, "mask": None})
    elif args.scenes_dir:
        root = Path(args.scenes_dir)
        for scene in sorted(p for p in root.iterdir() if p.is_dir()):
            if not ((scene / "im0.png").is_file() and (scene / "im1.png").is_file()):
                continue
            gt_dir = Path(args.gt_root) / scene.name if args.gt_root else scene
            gt = next((gt_dir / n for n in ["disp0GT.pfm", "disp0.pfm"] if (gt_dir / n).is_file()), None)
            mask = gt_dir / "mask0nocc.png" if (gt_dir / "mask0nocc.png").is_file() else None
            pairs.append({"name": scene.name, "left": scene / "im0.png", "right": scene / "im1.png", "gt": gt, "mask": mask})
    else:
        sys.exit("ERROR: give an input: --test_images, --left/--right, --left_dir/--right_dir or --scenes_dir (see --help)")
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        sys.exit("ERROR: no stereo pairs found")
    for p in pairs:
        p.setdefault("save_dir", None)
        p.setdefault("calib", None)
        for key in ("left", "right"):
            if not p[key].is_file():
                sys.exit(f"ERROR: file not found: {p[key]}")
    return pairs


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="FoundationStereo inference with model selection",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = ap.add_argument_group("model")
    g.add_argument("--model", default="zero_shot",
                   help="name(s) from models.json, comma separated, or 'all' (default: zero_shot)")
    g.add_argument("--ckpt", help="use this checkpoint file instead of --model (any {'model': state_dict} .pth)")
    g.add_argument("--adapter", help="LoRA adapter .pth; applied on --ckpt, or on the zero-shot model if no --ckpt")
    g.add_argument("--list", action="store_true", help="list the models in the registry and exit")
    g.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    g.add_argument("--repo_root", default=str(DEFAULT_REPO), help="FoundationStereo source folder")
    g.add_argument("--cfg", help="cfg.yaml (default: the one next to the zero-shot checkpoint)")

    g = ap.add_argument_group("input (choose one)")
    g.add_argument("--test_images", nargs="?", const=str(HERE / "test_images"),
                   help="folder with one sub-folder per pair (left.png + right.png, optional K.txt / gt / mask); "
                        "results are saved INTO each sub-folder. Without a value: inference/test_images")
    g.add_argument("--left"); g.add_argument("--right")
    g.add_argument("--gt", help="ground-truth disparity for the single pair (.png KITTI /256, .pfm, .npy)")
    g.add_argument("--mask", help="optional mask image; pixels == 255 are evaluated (e.g. Middlebury mask0nocc.png)")
    g.add_argument("--left_dir"); g.add_argument("--right_dir")
    g.add_argument("--gt_dir", help="ground truth folder with the same filenames as --left_dir")
    g.add_argument("--pattern", default="*", help="filename filter for --left_dir, e.g. '*_10.png' for KITTI")
    g.add_argument("--scenes_dir", help="folder of scene sub-folders containing im0.png / im1.png (Middlebury)")
    g.add_argument("--gt_root", help="for --scenes_dir: folder with <scene>/disp0GT.pfm and mask0nocc.png")
    g.add_argument("--limit", type=int, default=0, help="only the first N pairs")

    g = ap.add_argument_group("inference")
    g.add_argument("--iters", type=int, default=32, help="GRU iterations (32 = what all our evaluations used)")
    g.add_argument("--scale", type=float, default=1.0, help="downscale input by this factor (<=1) to save memory")
    g.add_argument("--hiera", action="store_true", help="hierarchical inference for >1K resolution images")
    g.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    g.add_argument("--fp32", action="store_true", help="disable fp16 autocast (fp16 is on by default on GPU)")

    g = ap.add_argument_group("output")
    g.add_argument("--out_dir", default=None,
                   help="default inference/outputs (with --test_images: maps go into each pair folder and "
                        "metrics.csv into the test_images folder)")
    g.add_argument("--save_pfm", action="store_true", help="also write disparity as .pfm")
    g.add_argument("--save_png16", action="store_true", help="also write KITTI-format 16-bit png (disp * 256)")
    g.add_argument("--focal", type=float, help="focal length in px (with --baseline: also save depth .npy in metres)")
    g.add_argument("--baseline", type=float, help="baseline in metres")
    g.add_argument("--no_vis", action="store_true", help="skip colour images / comparison grids")
    return ap.parse_args()


def main():
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    data, models, aliases = load_registry(registry_path)
    reg_dir = registry_path.parent
    base_found, base_expected = first_existing(models["zero_shot"]["path"], reg_dir)
    base_ckpt = base_found or base_expected
    cfg_path = Path(args.cfg) if args.cfg else base_ckpt.parent / "cfg.yaml"

    if args.list:
        print_model_list(models, reg_dir, base_ckpt)
        return

    # ---------- which models ----------
    runs = []  # (display name, full ckpt, adapter)
    if args.ckpt or args.adapter:
        # explicit files: --ckpt alone, --ckpt + --adapter, or --adapter alone (applied on the zero-shot base)
        ckpt = Path(args.ckpt) if args.ckpt else base_ckpt
        adapter = Path(args.adapter) if args.adapter else None
        for f in [ckpt, adapter]:
            if f is not None and not f.is_file():
                sys.exit(f"ERROR: file not found: {f}")
        runs.append(((adapter or ckpt).stem.replace("_lora_adapter", ""), ckpt, adapter))
    else:
        requested = list(models) if args.model.strip().lower() == "all" else \
            [m.strip() for m in args.model.split(",") if m.strip()]
        for req in requested:
            name = aliases.get(req.lower())
            if name is None:
                sys.exit(f"ERROR: unknown model '{req}'. Known: {', '.join(models)} (aliases: run --list)")
            full, adapter, full_expected, adapter_expected = model_files(models[name], reg_dir)
            if full:
                runs.append((name, full, None))  # full checkpoints already contain merged LoRA
            elif adapter and base_ckpt.is_file():
                print(f"[info] {name}: full checkpoint not found, using zero-shot base + LoRA adapter {adapter.name}")
                runs.append((name, base_ckpt, adapter))
            else:
                msg = f"{name}: checkpoint not found at {full_expected}"
                if adapter_expected:
                    msg += f" (nor adapter {adapter_expected})"
                if args.model.strip().lower() == "all":
                    print(f"[warn] {msg} - skipped")
                    continue
                sys.exit(f"ERROR: {msg}\n  Put the file there, or edit {registry_path}, or pass --ckpt <file>.")
    if not runs:
        sys.exit("ERROR: no model files found. Run with --list to see where they are expected.")
    if not cfg_path.is_file():
        sys.exit(f"ERROR: cfg.yaml not found at {cfg_path} (use --cfg)")

    # ---------- device ----------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            sys.exit("ERROR: --device cuda requested but CUDA is not available")
    fp16 = device.type == "cuda" and not args.fp32
    if not 0 < args.scale <= 1:
        sys.exit("ERROR: --scale must be in (0, 1]")

    pairs = collect_pairs(args)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.test_images:
        out_dir = Path(args.test_images)
    else:
        out_dir = HERE / "outputs"

    def out_file(p, model_name, suffix):
        # --test_images: <pair folder>/<model><suffix>; otherwise <out_dir>/<model>/<pair><suffix>
        if p["save_dir"] is not None:
            return p["save_dir"] / f"{model_name}{suffix}"
        folder = out_dir / model_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{p['name']}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FoundationStereo inference")
    print(f"  models   : {', '.join(r[0] for r in runs)}")
    print(f"  pairs    : {len(pairs)} ({sum(p['gt'] is not None for p in pairs)} with ground truth)")
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"  device   : {device}{gpu} | precision {'fp16 autocast' if fp16 else 'fp32'} | iters {args.iters}"
          f" | scale {args.scale} | hierarchical {args.hiera}")
    if args.test_images:
        print(f"  output   : saved inside each pair folder of {Path(args.test_images).resolve()}")
        for p in pairs:
            extra = [k for k in ("gt", "mask") if p[k]] + (["K.txt -> depth"] if p["calib"] else [])
            print(f"             {p['name']}: {p['left'].name} + {p['right'].name}"
                  + (f"  ({', '.join(extra)})" if extra else ""))
    else:
        print(f"  output   : {out_dir.resolve()}")
    print("=" * 100, flush=True)

    fs = import_foundation_stereo(args.repo_root)
    InputPadder = fs[2]

    gts = {p["name"]: (read_gt(p["gt"]) if p["gt"] else None) for p in pairs}
    masks = {p["name"]: (imageio.imread(p["mask"]) == 255 if p["mask"] else None) for p in pairs}
    metric_rows = []
    for gt_name, gt in gts.items():
        if gt is not None:
            print(f"  GT {gt_name}: {np.isfinite(gt).sum():,} valid px, range {np.nanmin(gt):.1f}-{np.nanmax(gt):.1f}"
                  + (f", mask {int(masks[gt_name].sum()):,} px" if masks[gt_name] is not None else ""))

    for run_number, (name, ckpt, adapter) in enumerate(runs, 1):
        print(f"\n[{run_number}/{len(runs)}] Loading {name} ...", flush=True)
        model = build_model(fs, cfg_path, ckpt, adapter, device, fp16, args.iters)
        for i, p in enumerate(pairs, 1):
            left, right = read_rgb(p["left"]), read_rgb(p["right"])
            if left.shape != right.shape:
                print(f"  [warn] {p['name']}: left {left.shape} and right {right.shape} differ - skipped")
                continue
            try:
                disp, sec = predict(model, InputPadder, left, right, device, fp16, args.iters, args.scale, args.hiera)
                used = "fp16" if fp16 else "fp32"
                if fp16 and not np.isfinite(disp).all():
                    print(f"  [warn] {p['name']}: non-finite output in fp16 (some GPUs, e.g. GTX 16xx) - rerunning in fp32")
                    disp, sec = predict(model, InputPadder, left, right, device, False, args.iters, args.scale, args.hiera)
                    used = "fp32"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                sys.exit(f"ERROR: GPU out of memory on {p['name']} {left.shape[1]}x{left.shape[0]}. "
                         f"Try --scale 0.5, --hiera, or --device cpu.")

            np.save(out_file(p, name, "_disp.npy"), disp)
            if args.save_pfm:
                write_pfm(out_file(p, name, "_disp.pfm"), disp)
            if args.save_png16:
                imageio.imwrite(out_file(p, name, "_disp_kitti.png"),
                                (np.nan_to_num(disp).clip(0, 255.99) * 256).astype(np.uint16))
            calib = (args.focal, args.baseline) if (args.focal and args.baseline) else p["calib"]
            depth = None
            if calib:
                with np.errstate(divide="ignore"):
                    depth = np.where(disp > 0, calib[0] * calib[1] / disp, np.inf).astype(np.float32)
                np.save(out_file(p, name, "_depth_m.npy"), depth)
            if not args.no_vis:
                gt = gts[p["name"]]
                vmin, vmax = disp_range([gt if gt is not None else disp])
                disp_color = colorize(disp, vmin, vmax)
                imageio.imwrite(out_file(p, name, "_disp.png"), disp_color)
                # left image | disparity side by side, like the official demo's vis.png
                imageio.imwrite(out_file(p, name, "_vis.png"), np.concatenate([left, disp_color], axis=1))
                if depth is not None:
                    finite = depth[np.isfinite(depth)]
                    if finite.size:
                        near, far = np.percentile(finite, 1), np.percentile(finite, 95)
                        # near = red, far = blue (same look as the disparity map); farther than 'far' is clipped
                        imageio.imwrite(out_file(p, name, "_depth.png"),
                                        colorize(-np.clip(depth, near, far), -far, -near))

            line = f"  [{i}/{len(pairs)}] {p['name']}: {left.shape[1]}x{left.shape[0]}, {sec:.2f}s ({used}), " \
                   f"disp {np.nanmin(disp):.1f}-{np.nanmax(disp):.1f} px"
            if depth is not None and np.isfinite(depth).any():
                line += f", depth median {np.median(depth[np.isfinite(depth)]):.2f} m"
            row = {"model": name, "pair": p["name"], "seconds": round(sec, 4), "precision": used}
            gt = gts[p["name"]]
            if gt is not None:
                if gt.shape != disp.shape:
                    print(f"  [warn] {p['name']}: GT {gt.shape} vs prediction {disp.shape} - metrics skipped")
                else:
                    m = compute_metrics(disp, gt, masks[p["name"]])
                    if m:
                        row.update(m)
                        line += f" | EPE {m['epe']:.3f}  D1 {m['d1']:.2f}%  BP-2 {m['bp2']:.2f}%"
            metric_rows.append(row)
            print(line, flush=True)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------- metrics ----------
    with_gt = [r for r in metric_rows if "epe" in r]
    if metric_rows:
        keys = ["model", "pair", "seconds", "precision", "valid_px", "epe", "d1", "bp1", "bp2", "bp3",
                "d1_bad_px", "bp2_bad_px"]
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metric_rows)
    if with_gt:
        print("\n" + "=" * 100)
        print("SUMMARY  (pixel-weighted EPE / D1 / BP-2 = same aggregation as the project notebooks; "
              "mean-image = average of per-image values)")
        print(f"{'model':<11} {'pairs':>5} {'EPE':>8} {'D1 %':>8} {'BP-2 %':>8} {'mean-img D1':>12} "
              f"{'mean-img BP-2':>14} {'s/pair':>7}")
        print("-" * 100)
        for name, *_ in runs:
            rows = [r for r in with_gt if r["model"] == name]
            if not rows:
                continue
            px = sum(r["valid_px"] for r in rows)
            print(f"{name:<11} {len(rows):>5} {sum(r['epe'] * r['valid_px'] for r in rows) / px:>8.4f} "
                  f"{100 * sum(r['d1_bad_px'] for r in rows) / px:>8.4f} "
                  f"{100 * sum(r['bp2_bad_px'] for r in rows) / px:>8.4f} "
                  f"{np.mean([r['d1'] for r in rows]):>12.4f} {np.mean([r['bp2'] for r in rows]):>14.4f} "
                  f"{np.mean([r['seconds'] for r in rows]):>7.2f}")
        print("Note: KITTI convention -> D1 on disp_occ (all) or disp_noc (noc); Middlebury -> BP-2 with mask0nocc.")

    # ---------- comparison grids (one per pair, all models stacked) ----------
    if not args.no_vis and len(runs) > 1:
        grid_dir = out_dir / "comparison"
        for p in pairs:
            disps = [(name, out_file(p, name, "_disp.npy")) for name, *_ in runs]
            disps = [(n, np.load(f)) for n, f in disps if f.is_file()]
            if not disps:
                continue
            gt = gts[p["name"]]
            vmin, vmax = disp_range([gt] if gt is not None else [d for _, d in disps])
            tiles = [label(read_rgb(p["left"]), f"left: {p['name']}")]
            if gt is not None and gt.shape == disps[0][1].shape:
                tiles.append(label(colorize(gt, vmin, vmax), "ground truth"))
            for n, d in disps:
                r = next((r for r in with_gt if r["model"] == n and r["pair"] == p["name"]), None)
                text = n + (f"  EPE {r['epe']:.2f}  D1 {r['d1']:.2f}%  BP-2 {r['bp2']:.2f}%" if r else "")
                tiles.append(label(colorize(d, vmin, vmax), text))
            h, w = tiles[0].shape[:2]
            tiles = [t if t.shape[:2] == (h, w) else cv2.resize(t, (w, h)) for t in tiles]
            if p["save_dir"] is not None:
                grid_path = p["save_dir"] / "comparison.png"
            else:
                grid_dir.mkdir(exist_ok=True)
                grid_path = grid_dir / f"{p['name']}_compare.png"
            imageio.imwrite(grid_path, np.concatenate(tiles, axis=0))
        print("\nComparison grids: "
              + ("comparison.png in each pair folder" if args.test_images else str(grid_dir.resolve())))

    if args.test_images:
        print(f"\nOutputs saved in each pair folder under {Path(args.test_images).resolve()}:")
        print("  <model>_disp.png (colour disparity), <model>_vis.png (left | disparity), <model>_disp.npy,")
        print("  <model>_depth.png + <model>_depth_m.npy (only if the folder has K.txt), comparison.png (if >1 model)")
        print(f"metrics.csv: {(out_dir / 'metrics.csv').resolve()}")
    else:
        print(f"Outputs: {out_dir.resolve()}  (per model: <pair>_disp.npy, _disp.png, _vis.png; metrics.csv)")


if __name__ == "__main__":
    main()
