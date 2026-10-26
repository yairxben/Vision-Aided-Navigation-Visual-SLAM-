# Experiment configs

Partial overrides of `generic_slam/config.py` defaults, passed with `--config`.
Reference: AKAZE default run (commit 08e9a77): loop closure 35.1 min,
RMSE with loops 2.03 m, 5 accepted loops.

| File | Change | Result on KITTI 00 | Status |
|---|---|---|---|
| `A_loop_ransac600.json` | Loop-closure RANSAC budget 3000 -> 600 | Same 5 loops, identical RMSE (2.03 m); loop closure 35.1 -> 20.5 min | **Adopted as default** |
| `B_loop_ransac600_radius3.json` | A + loop suppression radius 10 -> 3 keyframes | 10 loops, all real; RMSE 1.94 m, final 1.44 m, but max 4.39 -> 4.66 m | Not adopted |

Rationale for A: a pair can only pass with >= 30% supporters; at w = 0.3,
s = 4, RANSAC needs log(0.01)/log(1 - 0.3^4) ~ 566 attempts for 99%
confidence. The extra budget was spent only on hopeless (false) candidates.
