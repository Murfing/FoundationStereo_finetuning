"""
Classical (non-learned) stereo baseline: OpenCV Semi-Global Block Matching (SGBM), optionally plain Block Matching (BM).

Evaluated EXACTLY like the FoundationStereo experiments (same code as foundationstereo_e1_e3_lora.ipynb):
  * Middlebury : Eval3 trainingH (half resolution), 15 scenes, non-occluded pixels (mask0nocc == 255),
                 EPE and BP-2 (% pixels with error > 2 px); pixel-weighted ("bp2") and per-scene mean ("mean_scene_bp2").
  * KITTI 2015 : Protocol A test split = 160 images (np.random.default_rng(2025) permutation, images 40..199),
                 EPE and D1 (error > 3 px AND > 5 % of GT) on disp_occ ("all") and disp_noc ("noc");
                 pixel-weighted EPE, global D1, mean-image D1. (Protocol A only uses KITTI 2015 for val/test.)

SGBM settings are fixed in advance (standard OpenCV recipe), NOT tuned on the test images:
  blockSize 5, P1 = 8*3*5^2, P2 = 32*3*5^2, uniquenessRatio 10, speckleWindowSize 100, speckleRange 2,
  disp12MaxDiff 1, preFilterCap 63, mode MODE_SGBM (5 path directions). Disparity range: Middlebury = the scene's
  official ndisp from calib.txt (rounded up to a multiple of 16); KITTI = 256.
Pixels SGBM rejects (uniqueness / left-right check / speckle filter, left image border) are filled with the KITTI
devkit rule "background interpolation" (each hole gets the smaller of its left/right valid neighbours on the same row),
so the method is scored on 100 % of the pixels like FoundationStereo. The fraction it left empty is reported as "density".

Usage (run from the "Computer vision" folder; CPU only, no GPU needed):
    python inference/classical_baseline.py --datasets middlebury
    python inference/classical_baseline.py --datasets kitti --download_kitti
    python inference/classical_baseline.py --datasets middlebury,kitti --methods sgbm,bm

Output (default inference/outputs/classical_baseline/<method>/):
    summary.json            aggregates in the same format as the notebook results + reference numbers + settings
    summary.md              ready-to-copy table for the slides (SGBM vs zero-shot FoundationStereo vs E0 / E1)
    middlebury_rows.csv     per scene: valid/bad pixels, EPE, BP-1/2/4, density, seconds
    kitti_test_rows.csv     per image and mask (all / noc): valid/bad pixels, EPE, D1, density, seconds
    vis/                    left | GT | prediction | bad pixels (red) for every Middlebury scene and the first KITTI images
    disp/                   Middlebury disparity maps (.npy, float32, after hole filling)
"""
import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent                     # .../Computer vision/inference
REPO = HERE.parent                                           # repository root
KITTI15_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"

# Reference numbers from our FoundationStereo runs (docs/my_notes.md, Protocol A, run 1)
REFERENCE = {
    "FoundationStereo zero-shot": {"middlebury": {"epe": 0.3634, "bp2": 1.0987, "mean_scene_bp2": 1.1475},
                                   "kitti_test": {"all": {"epe": 0.8545, "d1": 2.8561, "mean_image_d1": 2.9054},
                                                  "noc": {"epe": 0.8351, "d1": 2.7549, "mean_image_d1": 2.8204}}},
    "E0 full fine-tune (Protocol A)": {"middlebury": {"epe": 0.6345, "bp2": 2.4225, "mean_scene_bp2": None},
                                       "kitti_test": {"all": {"epe": 0.7343, "d1": 1.8623, "mean_image_d1": 1.7978},
                                                      "noc": {"epe": 0.7152, "d1": 1.7146, "mean_image_d1": 1.6744}}},
    "E1 BN locked (Protocol A)": {"middlebury": {"epe": 0.6552, "bp2": 2.1074, "mean_scene_bp2": 2.2387},
                                  "kitti_test": {"all": {"epe": 0.7145, "d1": 1.4979, "mean_image_d1": None},
                                                 "noc": {"epe": None, "d1": 1.3625, "mean_image_d1": None}}},
}
# Protocol A indices that must come out of the split (from the E0 log) - guards against a different split
SPLIT_CHECK = {"val #1": (0, "000139"), "val #10": (9, "000044"), "test #1": (40, "000129"), "test #10": (49, "000008")}


# ----------------------------------------------------------------------------------------------- matchers
def make_matcher(method, num_disp):
    num_disp = int(np.ceil(num_disp / 16.0) * 16)
    if method == "sgbm":
        block = 5
        return cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                                     P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2, disp12MaxDiff=1,
                                     uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                     preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM)
    if method == "bm":
        bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
        bm.setUniquenessRatio(10)
        bm.setSpeckleWindowSize(100)
        bm.setSpeckleRange(2)
        bm.setDisp12MaxDiff(1)
        return bm
    raise ValueError(method)


def fill_holes(disp, valid):
    """KITTI devkit 'interpolateBackground': per row, each invalid pixel gets min(nearest valid left, nearest valid right)."""
    h, w = disp.shape
    cols = np.broadcast_to(np.arange(w), (h, w))
    left_idx = np.maximum.accumulate(np.where(valid, cols, -1), axis=1)                     # last valid at or left of x
    right_idx = np.minimum.accumulate(np.where(valid, cols, w)[:, ::-1], axis=1)[:, ::-1]    # first valid at or right of x
    rows = np.arange(h)[:, None]
    left_val = np.where(left_idx >= 0, disp[rows, np.clip(left_idx, 0, w - 1)], np.inf)
    right_val = np.where(right_idx < w, disp[rows, np.clip(right_idx, 0, w - 1)], np.inf)
    filled = np.minimum(left_val, right_val)
    filled[~np.isfinite(filled)] = 0.0                                                        # row without any match
    return np.where(valid, disp, filled).astype(np.float32)


def predict(method, left_bgr, right_bgr, num_disp):
    matcher = make_matcher(method, num_disp)
    if method == "bm":
        left_in, right_in = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (left_bgr, right_bgr))
    else:
        left_in, right_in = left_bgr, right_bgr
    start = time.perf_counter()
    raw = matcher.compute(left_in, right_in).astype(np.float32) / 16.0                        # fixed point -> pixels
    valid = raw >= 0                                                                            # invalid = minDisparity-1 = -1
    disp = fill_holes(raw, valid)
    return disp, float(valid.mean()), time.perf_counter() - start


# ----------------------------------------------------------------------------------------------- data
def read_pfm(path):
    with open(path, "rb") as file:
        if file.readline().strip() != b"Pf":
            raise ValueError(path)
        width, height = map(int, file.readline().split())
        scale = float(file.readline())
        values = np.fromfile(file, ("<" if scale < 0 else ">") + "f4")
    return np.flipud(values.reshape(height, width)).astype(np.float32)


def read_kitti_disp(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0


def middlebury_dirs(root):
    data = next((p for p in Path(root).rglob("MiddEval3") if (p / "trainingH").is_dir() and
                 any((p / "trainingH").glob("*/im0.png"))), None)
    gt = next((p for p in Path(root).rglob("MiddEval3") if any((p / "trainingH").glob("*/disp0GT.pfm"))), None)
    if data is None or gt is None:
        raise SystemExit(f"Middlebury trainingH data / GT not found under {root} (expects data-H and gt0-H)")
    return data / "trainingH", gt / "trainingH"


def ensure_kitti15(root, download, delete_zip):
    root = Path(root)
    found = [p.parent for p in root.rglob("image_2") if p.parent.name == "training" and (p.parent / "disp_occ_0").is_dir()]
    if found:
        return found[0]
    if not download:
        raise SystemExit(f"KITTI 2015 training data not found under {root}. Add --download_kitti (about 1.7 GB download).")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "data_scene_flow.zip"
    if not archive.exists():
        import urllib.request
        print(f"Downloading {KITTI15_URL} (about 1.7 GB) ...", flush=True)
        partial = archive.with_suffix(".zip.part")
        request = urllib.request.Request(KITTI15_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(archive)
    print("Extracting the folders needed (image_2, image_3, disp_occ_0, disp_noc_0) ...", flush=True)
    wanted = ("training/image_2/", "training/image_3/", "training/disp_occ_0/", "training/disp_noc_0/")
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if m.startswith(wanted) and m.endswith("_10.png")]
        z.extractall(root / "kitti2015", members)
    if delete_zip:
        archive.unlink()
    return ensure_kitti15(root, False, False)


def kitti15_protocol_a(training_root):
    """Same indexing and the same random draws as the notebooks -> identical Protocol A split."""
    indices = sorted({p.stem.split("_", 1)[0] for p in (training_root / "image_2").glob("*.png")})
    if len(indices) != 200:
        raise SystemExit(f"Expected 200 KITTI 2015 training pairs, found {len(indices)}")
    rng = np.random.default_rng(2025)
    k15_order = rng.permutation(len(indices))
    ordered = [indices[i] for i in k15_order]
    for label, (pos, expected) in SPLIT_CHECK.items():
        if ordered[pos] != expected:
            raise SystemExit(f"Protocol A split mismatch at {label}: {ordered[pos]} != {expected}")
    print("Protocol A split identical to E0 (val #1 000139, test #1 000129)")
    return {"val": ordered[:40], "test": ordered[40:]}


# ----------------------------------------------------------------------------------------------- visualisation
def colorize(disp, lo, hi):
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 18 + 15 * len(text)), 40), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_vis(path, left, gt, gt_valid, pred, bad, title):
    vals = gt[gt_valid]
    lo, hi = (np.percentile(vals, 1), np.percentile(vals, 99)) if vals.size else (0, max(pred.max(), 1))
    gt_img = colorize(np.where(gt_valid, gt, lo), lo, hi)
    gt_img[~gt_valid] = 0
    err = (cv2.cvtColor(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.5).astype(np.uint8)
    err[gt_valid & ~bad] = (0, 170, 0)
    err[gt_valid & bad] = (0, 0, 255)
    panels = [label(left, "left"), label(gt_img, "ground truth"), label(colorize(pred, lo, hi), title),
              label(err, "red = bad pixel, green = good")]
    top, bottom = np.hstack(panels[:2]), np.hstack(panels[2:])
    cv2.imwrite(str(path), np.vstack([top, bottom]))


# ----------------------------------------------------------------------------------------------- evaluation
def eval_middlebury(method, root, out):
    data_dir, gt_dir = middlebury_dirs(root)
    scenes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if len(scenes) != 15:
        raise SystemExit(f"Expected 15 Middlebury trainingH scenes, found {len(scenes)}")
    (out / "vis").mkdir(parents=True, exist_ok=True)
    (out / "disp").mkdir(parents=True, exist_ok=True)
    rows = []
    for number, scene in enumerate(scenes, start=1):
        left = cv2.imread(str(data_dir / scene / "im0.png"))
        right = cv2.imread(str(data_dir / scene / "im1.png"))
        calib = dict(line.strip().split("=", 1) for line in open(data_dir / scene / "calib.txt") if "=" in line)
        ndisp = int(calib["ndisp"])
        gt = read_pfm(gt_dir / scene / "disp0GT.pfm")
        mask = cv2.imread(str(gt_dir / scene / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255
        pred, density, seconds = predict(method, left, right, ndisp)
        valid = np.isfinite(pred) & np.isfinite(gt) & mask                    # identical to the notebook
        error = np.abs(pred[valid] - gt[valid])
        row = {"scene": scene, "ndisp": ndisp, "valid_pixels": int(valid.sum()), "bad_pixels": int((error > 2.0).sum()),
               "epe": float(error.mean()), "bp1": float((error > 1.0).mean() * 100), "bp2": float((error > 2.0).mean() * 100),
               "bp4": float((error > 4.0).mean() * 100), "density": round(density * 100, 2), "seconds": round(seconds, 2)}
        rows.append(row)
        np.save(out / "disp" / f"middlebury_{scene}.npy", pred)
        bad_full = np.zeros_like(valid)
        bad_full[valid] = error > 2.0
        save_vis(out / "vis" / f"middlebury_{scene}.png", left, gt, valid, pred, bad_full, method.upper())
        print(f"Middlebury {number:2d}/15 {scene:<12} EPE {row['epe']:7.3f}  BP-2 {row['bp2']:6.2f}%  "
              f"density {row['density']:5.1f}%  {seconds:5.1f}s", flush=True)
    total = sum(r["valid_pixels"] for r in rows)
    summary = {"epe": round(sum(r["epe"] * r["valid_pixels"] for r in rows) / total, 4),
               "bp2": round(100.0 * sum(r["bad_pixels"] for r in rows) / total, 4),
               "mean_scene_bp2": round(float(np.mean([r["bp2"] for r in rows])), 4),
               "mean_density": round(float(np.mean([r["density"] for r in rows])), 2),
               "valid_pixels": total}
    write_csv(out / "middlebury_rows.csv", rows)
    return summary


def eval_kitti(method, training_root, split, out, n_vis):
    rows = []
    (out / "vis").mkdir(parents=True, exist_ok=True)
    for number, index in enumerate(split, start=1):
        name = f"{index}_10.png"
        left = cv2.imread(str(training_root / "image_2" / name))
        right = cv2.imread(str(training_root / "image_3" / name))
        occ = read_kitti_disp(training_root / "disp_occ_0" / name)
        noc = read_kitti_disp(training_root / "disp_noc_0" / name)
        pred, density, seconds = predict(method, left, right, 256)
        for mask_name, gt in [("all", occ), ("noc", noc)]:
            valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)            # identical to the notebook
            error = np.abs(pred[valid] - gt[valid])
            bad = (error > 3.0) & (error / np.maximum(gt[valid], 1e-6) > 0.05)
            rows.append({"dataset": "kitti2015", "index": index, "mask": mask_name, "valid_pixels": int(valid.sum()),
                         "epe": float(error.mean()), "bad_pixels": int(bad.sum()), "d1": float(bad.mean() * 100.0),
                         "density": round(density * 100, 2), "seconds": round(seconds, 2)})
            if mask_name == "all" and number <= n_vis:
                bad_full = np.zeros_like(valid)
                bad_full[valid] = bad
                save_vis(out / "vis" / f"kitti_{index}.png", left, gt, valid, pred, bad_full, method.upper())
        if number == 1 or number % 10 == 0 or number == len(split):
            print(f"KITTI test {number:3d}/{len(split)} ({index})  D1-all {rows[-2]['d1']:6.2f}%  "
                  f"density {density * 100:5.1f}%  {seconds:4.1f}s", flush=True)
    out_summary = {}
    for mask_name in ["all", "noc"]:
        subset = [r for r in rows if r["mask"] == mask_name]
        total = sum(r["valid_pixels"] for r in subset)
        out_summary[mask_name] = {"epe": round(sum(r["epe"] * r["valid_pixels"] for r in subset) / total, 4),
                                  "d1": round(100.0 * sum(r["bad_pixels"] for r in subset) / total, 4),
                                  "mean_image_d1": round(float(np.mean([r["d1"] for r in subset])), 4)}
    out_summary["images"] = len(split)
    out_summary["mean_density"] = round(float(np.mean([r["density"] for r in rows if r["mask"] == "all"])), 2)
    write_csv(out / "kitti_test_rows.csv", rows)
    return out_summary


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def write_markdown(path, method, results):
    name = {"sgbm": "SGBM (OpenCV, classical)", "bm": "Block matching (OpenCV, classical)"}[method]
    lines = [f"# Classical baseline: {name}", "",
             "Lower is better. Middlebury: trainingH, 15 scenes, non-occluded. KITTI 2015: Protocol A test, 160 images.", "",
             "| Model | Middlebury BP-2 % | Middlebury EPE px | KITTI-15 D1-all % | KITTI-15 D1-noc % | KITTI-15 EPE px |",
             "|---|---|---|---|---|---|"]
    mb, kt = results.get("middlebury"), results.get("kitti_test")
    lines.append(f"| **{name}** | {fmt(mb and mb['bp2'])} | {fmt(mb and mb['epe'])} | {fmt(kt and kt['all']['d1'])} | "
                 f"{fmt(kt and kt['noc']['d1'])} | {fmt(kt and kt['all']['epe'])} |")
    for ref_name, ref in REFERENCE.items():
        m, k = ref["middlebury"], ref["kitti_test"]
        lines.append(f"| {ref_name} | {fmt(m['bp2'])} | {fmt(m['epe'])} | {fmt(k['all']['d1'])} | {fmt(k['noc']['d1'])} | "
                     f"{fmt(k['all']['epe'])} |")
    if mb:
        lines += ["", f"SGBM matched {mb['mean_density']:.1f} % of Middlebury pixels on average"
                  if method == "sgbm" else f"BM matched {mb['mean_density']:.1f} % of Middlebury pixels on average",
                  "(the rest were filled by background interpolation and scored like every other pixel)."]
    if kt:
        lines += [f"KITTI: {kt['mean_density']:.1f} % of pixels matched on average."]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="middlebury,kitti", help="comma list: middlebury, kitti")
    ap.add_argument("--methods", default="sgbm", help="comma list: sgbm, bm")
    ap.add_argument("--middlebury_root", default=str(REPO / "data" / "middlebury" / "raw"))
    ap.add_argument("--kitti_root", default=str(REPO / "data" / "kitti" / "raw"))
    ap.add_argument("--download_kitti", action="store_true", help="download KITTI 2015 (data_scene_flow.zip, ~1.7 GB)")
    ap.add_argument("--delete_zip", action="store_true", help="delete the KITTI zip after extracting the needed folders")
    ap.add_argument("--kitti_vis", type=int, default=10, help="save visualisations for the first N KITTI test images")
    ap.add_argument("--out_dir", default=str(HERE / "outputs" / "classical_baseline"))
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    kitti_root = split = None
    if "kitti" in datasets:
        kitti_root = ensure_kitti15(args.kitti_root, args.download_kitti, args.delete_zip)
        split = kitti15_protocol_a(kitti_root)["test"]

    for method in methods:
        out = Path(args.out_dir) / method
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "summary.json"
        results = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        print(f"\n===== {method.upper()} =====")
        if "middlebury" in datasets:
            results["middlebury"] = eval_middlebury(method, args.middlebury_root, out)
            m = results["middlebury"]
            print(f"Middlebury: BP-2 {m['bp2']:.4f}% (weighted), {m['mean_scene_bp2']:.4f}% (mean-scene), EPE {m['epe']:.4f}")
        if "kitti" in datasets:
            results["kitti_test"] = eval_kitti(method, kitti_root, split, out, args.kitti_vis)
            for mask in ["all", "noc"]:
                k = results["kitti_test"][mask]
                print(f"KITTI test {mask}: EPE {k['epe']:.4f}  global D1 {k['d1']:.4f}%  mean-image D1 {k['mean_image_d1']:.4f}%")
        results["settings"] = {
            "method": method, "opencv": cv2.__version__,
            "sgbm": "blockSize 5, P1 8*3*25, P2 32*3*25, uniqueness 10, speckle 100/2, disp12MaxDiff 1, preFilterCap 63, MODE_SGBM",
            "bm": "grayscale, blockSize 15, uniqueness 10, speckle 100/2, disp12MaxDiff 1",
            "num_disparities": "Middlebury: calib.txt ndisp rounded up to x16; KITTI: 256",
            "holes": "filled by background interpolation (KITTI devkit rule); density = % pixels matched before filling",
            "middlebury": "trainingH, 15 scenes, mask0nocc == 255, pixel-weighted",
            "kitti": "KITTI 2015 Protocol A test (160), rng 2025, disp_occ = all, disp_noc = noc"}
        results["reference_foundationstereo"] = REFERENCE
        summary_path.write_text(json.dumps(results, indent=2))
        write_markdown(out / "summary.md", method, results)
        print(f"\nSaved: {out}  (summary.json, summary.md, *_rows.csv, vis/, disp/)")


if __name__ == "__main__":
    main()
