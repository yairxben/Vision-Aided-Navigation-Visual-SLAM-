# Experiment configs

Partial overrides of `generic_slam/config.py` defaults, passed with `--config`.
Reference: AKAZE default run (commit 08e9a77): loop closure 35.1 min,
RMSE with loops 2.03 m, 5 accepted loops.

| File | Change | Result on KITTI 00 | Status |
|---|---|---|---|
| `A_loop_ransac600.json` | Loop-closure RANSAC budget 3000 -> 600 | Same 5 loops, identical RMSE (2.03 m); loop closure 35.1 -> 20.5 min | **Adopted as default** |
| `B_loop_ransac600_radius3.json` | A + loop suppression radius 10 -> 3 keyframes | 10 loops, all real; RMSE 1.94 m, final 1.44 m, but max 4.39 -> 4.66 m | Not adopted |
| `C_temporal_stereo_rows.json` | Temporal matching on stereo-inlier rows only | Tracking 21.2 -> 15.5 min (run 38 -> 30 min); supporters +1.1%; RMSE within seed noise (see below) | **Adopted as default** |
| `D_stereo_row_band_mutual.json` | Stereo nearest neighbour inside the row band, one-to-one | 60 frames: stereo rows +15%, supporters +6% (84.8%); matcher 20 -> 38 ms/frame (slower in Python) | Not evaluated on full sequence |
| `E_stereo_row_band_ratio.json` | Row band + ratio test 0.8 | 60 frames: stereo rows +10%, supporters +1%; matcher 20 -> 32 ms/frame | Not evaluated on full sequence |

Rationale for A: a pair can only pass with >= 30% supporters; at w = 0.3,
s = 4, RANSAC needs log(0.01)/log(1 - 0.3^4) ~ 566 attempts for 99%
confidence. The extra budget was spent only on hopeless (false) candidates.

## Judging result-changing experiments: use two seeds

Any change to the matches shifts the RANSAC random sequence, and drift then
accumulates differently. Changing only `visual_odometry.seed` (0 -> 1) on the
baseline moved RMSE with loops 2.03 -> 2.13 m and PnP RMSE 29.24 -> 27.05 m.
So run every result-changing experiment with seeds 0 and 1 and compare means.

| Run (RMSE m) | with loops | Bundle, no loops | PnP |
|---|---|---|---|
| baseline, seed 0 / 1 | 2.03 / 2.13 | 19.20 / 19.09 | 29.24 / 27.05 |
| C, seed 0 / 1 | 2.13 / 2.07 | 19.66 / 19.23 | 29.73 / 29.39 |
