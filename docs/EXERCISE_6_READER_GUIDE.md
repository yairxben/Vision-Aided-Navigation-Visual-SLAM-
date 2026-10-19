# Exercise 6 Reader Guide

## Purpose

Exercise 5 supplies 355 optimized relative poses between consecutive
keyframes and the corresponding conditional endpoint covariances. Exercise 6
turns that compact evidence into an anchored 356-camera Pose Graph, optimizes
it, computes pose Marginals, and writes the no-Loop-Closure trajectory,
uncertainty, and error evidence needed by the course report.

No Exercise 5 Bundle graph, landmark, stereo observation, or TrackingDB is
loaded by Exercise 6.

## Algorithm flow and file responsibilities

```text
diagnostics/ex5_clean_final
    -> load consecutive relative poses and covariances
    -> build the anchored Pose Graph
    -> compose the initial trajectory
    -> optimize with default GTSAM Levenberg-Marquardt
    -> compute standard GTSAM Marginals
    -> serialize arrays, metrics, tables, and figures
    -> diagnostics/ex6_clean_final
```

- `PoseGraph.py` validates the selected sealed Exercise 5 root, loads the
  ordered endpoint measurements, constructs the graph and initial `Values`,
  runs optimization, and constructs Marginals.
- `ex6.py` is the thin runner and command-line interface. Its scientific path
  follows the flow above and applies the established numerical checks.
- `exercise6_reporting.py` serializes the already-computed values and creates
  E6 tables and figures. It contains no second Pose Graph implementation.
- `test_ex6_pose_graph.py` protects the synthetic chain, clean Exercise 5
  contract, graph structure, factor ordering, conventions, initialization,
  and, when the optional historical E5 fixture is available,
  clean-versus-historical endpoint equivalence.

The graph contains 356 `Pose3` variables, one identity
`PriorFactorPose3`, and 355 consecutive `BetweenFactorPose3` factors. It has
no landmarks, reprojection factors, or Loop Closure factors.

## Scientific conventions

- Camera keys are `gtsam.symbol("c", frame_id)` using actual KITTI FrameIds,
  not keyframe ordinals.
- Each stored measurement is `start_pose.between(end_pose)` in direction
  `T_Cstart_from_Cend`.
- Each factor consumes the raw
  `conditional_end_given_start_covariance` directly through
  `gtsam.noiseModel.Gaussian.Covariance`.
- The GTSAM `Pose3` tangent order is
  `[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]`, with rotation in radians
  and translation in metres.
- The first camera is identity and has exactly one Candidate-A
  `Diagonal.Sigmas` prior. No other prior is added.
- Initialization starts at identity and repeatedly composes the same stored
  consecutive measurements used by the factors.

Because the graph is only this anchored consecutive chain, the composed
initial trajectory already reconstructs every measurement to floating-point
precision. Near-zero graph error and near-zero optimization movement are
therefore expected. Optimization is still executed once with the default
`gtsam.LevenbergMarquardtOptimizer`; it is not skipped. Marginals are then
constructed as `gtsam.Marginals(graph, optimized_values)` and saved without
scaling, regularization, or basis changes.

## Running the clean canonical path

Run from the repository root with the repository environment:

```bash
.venv/bin/python ex6.py \
    --input-root diagnostics/ex5_clean_final \
    --output-root diagnostics/ex6_clean_final
```

Both roots are explicit. The input must identify itself as canonical and
`complete-and-sealed` through its own `manifest.json` and embedded
configuration contract. The output directory must not already exist.

After the scientific run succeeds, create the two accepted presentation-only
covariance `x0.01` copies without rebuilding or reoptimizing the Pose Graph:

```bash
.venv/bin/python ex6.py \
    --input-root diagnostics/ex5_clean_final \
    --output-root diagnostics/ex6_clean_final \
    --create-visualization-x0.01-only
```

This option reads the saved poses and raw Marginals, adds two visualization
files, and records their presentation-only multiplier in `summary.json`. It
does not modify factor covariances, saved covariance arrays, raw figures,
Marginals, metrics, or optimized values.

## Final code and requirement traceability

Line numbers below describe the accepted E6 source layout for this clean run.

| Stage / requirement | File | Function | Line |
|---|---|---|---:|
| Sealed E5 input loading and contract validation | `PoseGraph.py` | `load_canonical_pose_graph_inputs` | 109 |
| Pose Graph construction and ordered factor creation | `PoseGraph.py` | `PoseGraph.create_factor_graph` | 471 |
| Identity anchor and composed initial trajectory | `PoseGraph.py` | `PoseGraph.create_factor_graph` | 471 |
| Default Levenberg-Marquardt optimization | `PoseGraph.py` | `PoseGraph.optimize` | 547 |
| Standard GTSAM Marginals | `PoseGraph.py` | `PoseGraph.compute_marginals` | 563 |
| Clean execution sequence and numerical checks | `ex6.py` | `run_canonical_pose_graph`; `_validate_optimization` | 43; 103 |
| Scientific array and summary serialization | `exercise6_reporting.py` | `write_pose_graph_outputs` | 456 |
| Absolute location/rotation error table and figures | `exercise6_reporting.py` | `write_report_ready_absolute_errors` | 211 |
| Initial and optimized top-down Pose Graph plots | `exercise6_reporting.py` | `plot_pose_graph` | 139 |
| First-Bundle covariance plot | `exercise6_reporting.py` | `plot_first_bundle_camera_covariances` | 103 |
| Pose Graph Marginals/covariance plot | `exercise6_reporting.py` | `plot_pose_graph_with_marginals` | 158 |
| Visualization-only `x0.01` copies | `exercise6_reporting.py` | `create_visualization_only_covariance_figures` | 355 |
| Focused scientific and input-equivalence tests | `test_ex6_pose_graph.py` | `SyntheticPoseGraphChainTest`; `CanonicalPoseGraphContractTest` | 33; 79 |

## Figure generators

| Output figure | Generator |
|---|---|
| `first_bundle_camera_covariances_3d.png` | `exercise6_reporting.plot_first_bundle_camera_covariances` |
| `first_bundle_camera_covariances_3d_visualization_x0.01.png` | `exercise6_reporting.create_visualization_only_covariance_figures` calling `plot_first_bundle_camera_covariances` |
| `initial_pose_graph.png` | `exercise6_reporting.plot_pose_graph` |
| `optimized_pose_graph.png` | `exercise6_reporting.plot_pose_graph` |
| `optimized_pose_graph_with_marginals.png` | `exercise6_reporting.plot_pose_graph_with_marginals` |
| `optimized_pose_graph_with_marginals_visualization_x0.01.png` | `exercise6_reporting.create_visualization_only_covariance_figures` calling `plot_pose_graph_with_marginals` |
| `pose_graph_ground_truth_trajectory.png` | `exercise6_reporting.write_report_ready_absolute_errors` |
| `pose_graph_absolute_errors.png` | `exercise6_reporting.write_report_ready_absolute_errors` |

The clean evidence retains the no-Loop-Closure trajectory, absolute X/Y/Z
and location-norm errors, absolute rotation error, and all 356 raw `Pose3`
marginal covariances. Exercise 7 will later consume this clean E6 baseline;
that handoff is outside the Exercise 6 run described here.
