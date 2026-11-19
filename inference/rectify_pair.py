"""
Rectify a stereo pair taken with one handheld camera (e.g. two phone photos a few cm apart).

    python inference/rectify_pair.py <left image> <right image> [--focal PX] [--out_dir DIR]
    python inference/rectify_pair.py inference/test_images/deepak/left.jpeg inference/test_images/deepak/right.jpeg

IMPORTANT: give the photo taken from the LEFT position first. The script keeps that order (it cannot reliably detect a
swap when the camera also turned between the shots - see "depth-order check" below).

Output (default: sibling folder "<input folder>_rectified", so `infer.py --test_images` picks it up):
  left.png, right.png    rectified pair (lossless PNG) -> feed these to infer.py
  rectify_info.txt       focal length, camera rotation/translation, before/after alignment numbers

Method:
  1. SIFT matches between the photos (RANSAC-filtered).
  2. Focal length: --focal if given, else estimated (the value with the most consistent matches, 0.5x-1.6x image size).
  3. Relative camera pose from the essential matrix; cv2.stereoRectify rotates both views virtually so that matching
     points lie on the same row.
  4. The turn of the camera adds an unknown CONSTANT horizontal offset to all disparities. The right image is shifted
     so that the smallest disparity is ~6 px (FoundationStereo needs disparities >= 0); relative depth is unaffected.
  5. Crop to the area valid in both images, then verify: |dy| < ~2 px and depth order consistent with the raw photos.
"--method hartley": projective rectification (cv2.stereoRectifyUncalibrated) instead of steps 2-3.

Limits: unknown baseline and unknown offset -> disparity is RELATIVE (near/far order and shape), not metric; no depth in
metres. The scene must not move between the two photos. Best results: slide the phone sideways 5-10 cm, don't turn it.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def match(L, R, max_features=10000):
    sift = cv2.SIFT_create(max_features)
    kl, dl = sift.detectAndCompute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), None)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), None)
    if dl is None or dr is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pairs = cv2.BFMatcher().knnMatch(dl, dr, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    pl = np.float32([kl[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pr = np.float32([kr[m.trainIdx].pt for m in good]).reshape(-1, 2)
    if len(pl) >= 15:
        _, mask = cv2.findFundamentalMat(pl, pr, cv2.FM_RANSAC, 1.0, 0.999)
        if mask is not None:
            keep = mask.ravel().astype(bool)
            pl, pr = pl[keep], pr[keep]
    return pl, pr


def alignment(pl, pr):
    dy, dx = pl[:, 1] - pr[:, 1], pl[:, 0] - pr[:, 0]
    return {"matches": int(len(dy)), "dy_med_abs": round(float(np.median(np.abs(dy))), 2),
            "dy_p90": round(float(np.percentile(np.abs(dy), 90)), 2),
            "dx_min": round(float(dx.min()), 1), "dx_med": round(float(np.median(dx)), 1),
            "dx_max": round(float(dx.max()), 1), "dx_neg_pct": round(float(100 * np.mean(dx < -3)), 1)}


def pose_rectification(L, pl, pr, focal):
    h, w = L.shape[:2]
    candidates = [focal] if focal else list(np.linspace(0.5, 1.6, 23) * max(w, h))
    best = None
    for f in candidates:
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        E, m = cv2.findEssentialMat(pl, pr, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            continue
        n, Rm, t, _ = cv2.recoverPose(E, pl, pr, K, mask=m.copy())
        if best is None or n > best[0]:
            best = (n, f, K, Rm, t)
    if best is None:
        return None
    n, f, K, Rm, t = best
    rot = np.degrees(cv2.Rodrigues(Rm)[0].ravel())
    t = t.ravel()
    info = {"method": "pose", "focal_px": round(float(f), 1), "pose_inliers": int(n),
            "rotation_deg_xyz": [round(float(a), 2) for a in rot],
            "translation_dir_xyz": [round(float(a), 3) for a in t]}
    print(f"  focal ~{f:.0f} px ({'given' if focal else 'estimated'}), {n} pose inliers")
    print(f"  camera turned between shots by about x {rot[0]:+.1f}, y {rot[1]:+.1f}, z {rot[2]:+.1f} deg")
    dist = np.zeros(5)
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K, dist, K, dist, (w, h), Rm, t, flags=cv2.CALIB_ZERO_DISPARITY,
                                                alpha=-1)
    maps = [cv2.initUndistortRectifyMap(K, dist, Rr, P, (w, h), cv2.CV_32FC1) for Rr, P in [(R1, P1), (R2, P2)]]
    warp = [lambda img, m=m, interp=cv2.INTER_LINEAR: cv2.remap(img, m[0], m[1], interp) for m in maps]
    pts = [lambda p, Rr=Rr, P=P: cv2.undistortPoints(p.reshape(-1, 1, 2), K, dist, R=Rr, P=P).reshape(-1, 2)
           for Rr, P in [(R1, P1), (R2, P2)]]
    return warp, pts, info


def hartley_rectification(L, pl, pr):
    h, w = L.shape[:2]
    F, _ = cv2.findFundamentalMat(pl, pr, cv2.FM_8POINT)
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pl, pr, F, (w, h))
    if not ok:
        return None
    warp = [lambda img, H=H, interp=cv2.INTER_LINEAR: cv2.warpPerspective(img, H, (w, h), flags=interp)
            for H in (H1, H2)]
    pts = [lambda p, H=H: cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
           for H in (H1, H2)]
    return warp, pts, {"method": "hartley"}


def shift_x(img, s):
    """Translate an image to the right by s pixels (s may be negative), zero fill."""
    M = np.float32([[1, 0, s], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def crop_valid(valid):
    rows = np.where(valid.mean(1) > 0.5)[0]
    if rows.size < 50:
        return None
    cols = np.where(valid[rows].mean(0) > 0.98)[0]
    if cols.size < 50:
        return None
    x0, x1 = cols.min(), cols.max() + 1
    rows = np.where(valid[:, x0:x1].mean(1) > 0.98)[0]
    if rows.size < 50:
        return None
    return x0, x1, rows.min(), rows.max() + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", help="photo taken from the LEFT position")
    ap.add_argument("right", help="photo taken from the RIGHT position")
    ap.add_argument("--out_dir", help="default: <folder of the left image>_rectified")
    ap.add_argument("--focal", type=float, help="focal length in pixels, if known (otherwise estimated)")
    ap.add_argument("--method", choices=["pose", "hartley"], default="pose")
    args = ap.parse_args()

    left_path, right_path = Path(args.left), Path(args.right)
    L, R = cv2.imread(str(left_path)), cv2.imread(str(right_path))
    if L is None or R is None:
        raise SystemExit("ERROR: could not read the images")
    if L.shape != R.shape:
        raise SystemExit("ERROR: images must have the same size (same camera, same zoom)")
    out_dir = Path(args.out_dir) if args.out_dir else left_path.parent.with_name(left_path.parent.name + "_rectified")

    pl, pr = match(L, R)
    if len(pl) < 30:
        raise SystemExit(f"ERROR: only {len(pl)} consistent matches - too few to rectify")
    before = alignment(pl, pr)
    print(f"BEFORE: rows off by |dy| median {before['dy_med_abs']} px (90%: {before['dy_p90']})")

    print(f"Rectifying ({args.method}) ...")
    rect = pose_rectification(L, pl, pr, args.focal) if args.method == "pose" else hartley_rectification(L, pl, pr)
    if rect is None:
        raise SystemExit("ERROR: rectification failed (degenerate geometry)")
    warp, pts, info = rect
    Lw, Rw = warp[0](L), warp[1](R)
    ones = np.full(L.shape[:2], 255, np.uint8)
    vL, vR = warp[0](ones, interp=cv2.INTER_NEAREST) > 0, warp[1](ones, interp=cv2.INTER_NEAREST) > 0

    # constant horizontal offset: shift the right image so the smallest disparity is ~6 px
    ql, qr = pts[0](pl), pts[1](pr)
    dx = ql[:, 0] - qr[:, 0]
    shift = float(np.percentile(dx, 1) - 6.0)   # farthest points end up at ~6 px
    Rw = shift_x(Rw, shift)
    vR = shift_x(vR.astype(np.uint8) * 255, shift) > 127
    qr = qr + np.array([shift, 0])
    box = crop_valid(vL & vR)
    if box is None:
        raise SystemExit("ERROR: the rectified images barely overlap - retake the photos (move sideways, don't turn)")
    x0, x1, y0, y1 = box
    Lc, Rc = Lw[y0:y1, x0:x1], Rw[y0:y1, x0:x1]

    # depth-order check: relative parallax must keep its order (rotation only adds a ~constant offset)
    raw_dx = pl[:, 0] - pr[:, 0]
    new_dx = ql[:, 0] - qr[:, 0]
    order_corr = float(np.corrcoef(raw_dx, new_dx)[0, 1])

    vl, vr = match(Lc, Rc)
    after = alignment(vl, vr) if len(vl) >= 15 else None
    if after is None:
        raise SystemExit("ERROR: could not verify the result (too few matches after rectification)")
    print(f"AFTER : rows off by |dy| median {after['dy_med_abs']} px (90%: {after['dy_p90']}); "
          f"disparity {after['dx_min']:+.0f}..{after['dx_max']:+.0f} px; size {Lc.shape[1]}x{Lc.shape[0]} "
          f"(input {L.shape[1]}x{L.shape[0]})")
    print(f"depth-order check: correlation of parallax before/after = {order_corr:+.2f} (should be clearly positive)")
    rows_ok = after["dy_p90"] <= 2.0 and after["dx_neg_pct"] <= 5
    if rows_ok and order_corr > 0.3:
        verdict = "RECTIFIED - usable for FoundationStereo (relative disparity only, no metric depth)"
    elif rows_ok:
        verdict = "rows aligned, but depth order could not be confirmed - treat the result with caution"
    else:
        verdict = "NOT well rectified - retake the photos (slide sideways, don't turn, static scene)"
    print("VERDICT:", verdict)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "left.png"), Lc)
    cv2.imwrite(str(out_dir / "right.png"), Rc)
    with open(out_dir / "rectify_info.txt", "w") as f:
        f.write(f"source left : {left_path.resolve()}\nsource right: {right_path.resolve()}\n{info}\n")
        f.write(f"right image shifted by {shift:+.1f} px; crop x {x0}:{x1}, y {y0}:{y1}\n")
        f.write(f"before: {before}\nafter : {after}\nparallax order correlation: {order_corr:+.3f}\n")
        f.write(f"verdict: {verdict}\nDisparity is relative (unknown baseline and offset): no depth in metres.\n")
    print(f"\nSaved: {out_dir / 'left.png'}\n       {out_dir / 'right.png'}\n       {out_dir / 'rectify_info.txt'}")


if __name__ == "__main__":
    main()
