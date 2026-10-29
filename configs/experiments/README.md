# Experiment configs

Partial overrides of `generic_slam/config.py` defaults, passed with `--config`.
Reference: AKAZE default run (commit 08e9a77): loop closure 35.1 min,
RMSE with loops 2.03 m, 5 accepted loops.

| File | Change | Result on KITTI 00 | Status |
|---|---|---|---|
| `A_loop_ransac600.json` | Loop-closure RANSAC budget 3000 -> 600 | Same 5 loops, identical RMSE (2.03 m); loop closure 35.1 -> 20.5 min | **Adopted as default** |
| `B_loop_ransac600_radius3.json` | A + loop suppression radius 10 -> 3 keyframes | 10 loops, all real; RMSE 1.94 m, final 1.44 m, but max 4.39 -> 4.66 m | Not adopted |
| `C_temporal_stereo_rows.json` | Temporal matching on stereo-inlier rows only | Tracking 21.2 -> 15.5 min (run 38 -> 30 min); supporters +1.1%; RMSE within seed noise (see below) | **Adopted as default** |
| `D_stereo_row_band_mutual.json` | Stereo nearest neighbour inside the row band, one-to-one | 60 frames: stereo rows +15%, supporters +6% (84.8%); matcher 20 -> 38 ms/frame (slower in Python). Full, 2 seeds: RMSE with loops 2.13/2.07 -> 1.83/1.84 m (-12%, first loop now closes 1.65 m apart instead of 9.9 m); Bundle and PnP RMSE within noise | Candidate; test applying it in Loop Closure only |
| `E_stereo_row_band_ratio.json` | Row band + ratio test 0.8 | 60 frames: stereo rows +10%, supporters +1%; matcher 20 -> 32 ms/frame | Not evaluated on full sequence |
| `F_orb3000.json` | ORB, 3000 features, instead of AKAZE | Full, 2 seeds: RMSE with loops 2.13/2.07 -> 3.42/3.27 m (+60%); Bundle 19.4 -> 34.6 m; PnP 29.6 -> 79 m; supporters 84% -> 67% of consensus; 4 loops instead of 5; tracking per frame about -50% | Not adopted (accuracy) |
| `G_akaze_threshold_0005.json` | AKAZE threshold 0.0002 -> 0.0005 | Full, 2 seeds: RMSE with loops 2.13/2.07 -> 2.22/2.33 m (+8%); Bundle 19.4 -> 20.7 m; PnP within noise; same 5 loops; stereo matches -29%; tracking per frame about -38% (runs in parallel, needs a solo timing) | Not adopted (accuracy), possible speed option |

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
| D, seed 0 / 1 | 1.83 / 1.84 | 19.79 / 19.65 | within noise |
| F (ORB), seed 0 / 1 | 3.42 / 3.27 | 34.91 / 34.22 | 79.70 / 78.20 |
| G (AKAZE 0.0005), seed 0 / 1 | 2.22 / 2.33 | 20.51 / 20.87 | 29.66 / 30.83 |
