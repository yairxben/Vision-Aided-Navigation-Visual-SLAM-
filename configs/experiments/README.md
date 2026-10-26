# Experiment configs

Partial overrides of `generic_slam/config.py` defaults, passed with `--config`.
Compare results against the AKAZE default run (commit 08e9a77): 90 min,
loop-closure 35.1 min, RMSE with loops 2.03 m, 5 accepted loops.

| File | Change | Rationale |
|---|---|---|
| `A_loop_ransac600.json` | Loop-closure RANSAC budget 3000 -> 600 | A pair can only pass with >= 30% supporters; at w = 0.3, s = 4 RANSAC needs log(0.01)/log(1 - 0.3^4) ~ 566 attempts for 99% confidence. The extra budget was spent only on hopeless (false) candidates. |
| `B_loop_ransac600_radius3.json` | A + loop suppression radius 10 -> 3 keyframes | Only 1-2 loops were kept per revisit; 249 suppressed pairs were true revisits within 20 m. More loop constraints per revisit. |
