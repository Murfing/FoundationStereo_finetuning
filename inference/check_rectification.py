"""
Check whether a left/right image pair is a rectified stereo pair (what FoundationStereo expects).

    python inference/check_rectification.py <left image> <right image>
    python inference/check_rectification.py inference/test_images/deepak/left.jpeg inference/test_images/deepak/right.jpeg

The script matches SIFT features between the two images and keeps only the geometrically consistent matches (RANSAC on
the fundamental matrix). For a rectified pair:
  - vertical offset  dy = y_left - y_right  is ~0 for every match (|dy| < ~1-2 px),
  - horizontal offset dx = x_left - x_right is >= 0 (a point appears further LEFT in the right image),
  - epipolar lines are horizontal.
Writes <left>_rectcheck.png next to the left image: matches drawn on top of each other, coloured by |dy|.
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    left_path, right_path = Path(sys.argv[1]), Path(sys.argv[2])
    L, R = cv2.imread(str(left_path)), cv2.imread(str(right_path))
    if L is None or R is None:
        sys.exit("ERROR: could not read one of the images")
    print(f"left  {left_path.name}: {L.shape[1]}x{L.shape[0]}")
    print(f"right {right_path.name}: {R.shape[1]}x{R.shape[0]}")
    if L.shape != R.shape:
        print("FAIL: the images have different sizes - not a rectified pair")

    sift = cv2.SIFT_create(8000)
    kl, dl = sift.detectAndCompute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), None)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), None)
    matches = cv2.BFMatcher().knnMatch(dl, dr, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        sys.exit(f"Only {len(good)} feature matches - images too different or too little texture to check")
    pl = np.float32([kl[m.queryIdx].pt for m in good])
    pr = np.float32([kr[m.trainIdx].pt for m in good])
    F, inl = cv2.findFundamentalMat(pl, pr, cv2.FM_RANSAC, 1.0, 0.999)
    inl = inl.ravel().astype(bool)
    pl, pr = pl[inl], pr[inl]
    dy = pl[:, 1] - pr[:, 1]
    dx = pl[:, 0] - pr[:, 0]
    print(f"\n{len(good)} matches, {inl.sum()} geometrically consistent (RANSAC)")
    print(f"vertical offset   dy = y_left - y_right : median {np.median(dy):+.1f} px, "
          f"|dy| median {np.median(np.abs(dy)):.1f} px, 90th pct {np.percentile(np.abs(dy), 90):.1f} px, "
          f"range {dy.min():+.0f}..{dy.max():+.0f}")
    print(f"horizontal offset dx = x_left - x_right : median {np.median(dx):+.1f} px, "
          f"{100 * np.mean(dx < -1):.0f}% negative, range {dx.min():+.0f}..{dx.max():+.0f}")

    # for a rectified pair F ~ [[0,0,0],[0,0,-1],[0,1,0]] (up to scale/sign); epipoles at infinity along x
    _, _, vt = np.linalg.svd(F)
    e = vt[-1]
    epipole = "at infinity" if abs(e[2]) < 1e-6 else f"at ({e[0] / e[2]:.0f}, {e[1] / e[2]:.0f}) px"
    print(f"epipole of the left image: {epipole} (rectified: at infinity, horizontal direction)")

    rows_ok = np.percentile(np.abs(dy), 90) <= 2.0
    order_ok = np.mean(dx < -1) <= 0.05
    swapped = np.mean(dx > 1) <= 0.05 and rows_ok
    print("\nVERDICT:")
    if rows_ok and order_ok:
        print("  RECTIFIED - rows line up and left/right order is correct. Fine for FoundationStereo.")
    elif swapped:
        print("  Rows line up but LEFT and RIGHT ARE SWAPPED - exchange the two files.")
    elif rows_ok:
        print(f"  Rows line up, but {100 * np.mean(dx < -1):.0f}% of points move RIGHT in the right image "
              "(should move LEFT): left/right order looks wrong.")
    else:
        print(f"  NOT RECTIFIED - the same point is on different rows (90% of matches off by up to "
              f"{np.percentile(np.abs(dy), 90):.0f} px; must be <~2 px).")
        print("  (Left/right order cannot be judged from these raw shifts: if the camera also turned between the")
        print("   shots, every point gets the same extra sideways shift. Rectify first: inference/rectify_pair.py)")
        print("  FoundationStereo results on this pair will be meaningless.")

    # visualisation: blend of both images, matches coloured green (|dy|<=2) -> red (|dy|>=20)
    h, w = L.shape[:2]
    Rr = cv2.resize(R, (w, h)) if R.shape != L.shape else R
    vis = cv2.addWeighted(L, 0.5, Rr, 0.5, 0)
    sel = np.random.default_rng(0).choice(len(pl), min(300, len(pl)), replace=False)
    for i in sel:
        t = float(np.clip((abs(dy[i]) - 2) / 18, 0, 1))
        color = (0, int(255 * (1 - t)), int(255 * t))
        cv2.line(vis, tuple(map(int, pl[i])), tuple(map(int, pr[i])), color, 2)
        cv2.circle(vis, tuple(map(int, pl[i])), 3, color, -1)
    for y in range(0, h, max(40, h // 20)):
        cv2.line(vis, (0, y), (w, y), (255, 255, 0), 1)
    out = left_path.with_name(left_path.stem + "_rectcheck.png")
    cv2.imwrite(str(out), vis)
    print(f"\nVisual check: {out}  (lines = how each matched point moves; rectified -> all lines horizontal, green)")


if __name__ == "__main__":
    main()
