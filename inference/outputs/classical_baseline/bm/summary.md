# Classical baseline: Block matching (OpenCV, classical)

Lower is better. Middlebury: trainingH, 15 scenes, non-occluded. KITTI 2015: Protocol A test, 160 images.

| Model | Middlebury BP-2 % | Middlebury EPE px | KITTI-15 D1-all % | KITTI-15 D1-noc % | KITTI-15 EPE px |
|---|---|---|---|---|---|
| **Block matching (OpenCV, classical)** | 29.03 | 8.62 | 23.19 | 22.16 | 4.23 |
| FoundationStereo zero-shot | 1.10 | 0.36 | 2.86 | 2.75 | 0.85 |
| E0 full fine-tune (Protocol A) | 2.42 | 0.63 | 1.86 | 1.71 | 0.73 |
| E1 BN locked (Protocol A) | 2.11 | 0.66 | 1.50 | 1.36 | 0.71 |

BM matched 50.9 % of Middlebury pixels on average
(the rest were filled by background interpolation and scored like every other pixel).
KITTI: 33.3 % of pixels matched on average.
