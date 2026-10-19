# Stereo Visual SLAM Pipeline — KITTI

A stereo Visual SLAM pipeline developed for the **Vision Aided Navigation** course, from stereo image correspondences and frame-to-frame Visual Odometry to local Bundle Adjustment, a global Pose Graph, and Loop Closure.

The project is evaluated on **KITTI Odometry Sequence 00** and uses **OpenCV** for feature-based geometry and **GTSAM** for nonlinear factor-graph optimization.

### Pipeline

**Stereo Images → Stereo Matching & Triangulation → PnP + RANSAC Visual Odometry → Multi-Frame Tracking Database → Sliding-Window Bundle Adjustment → Pose Graph → Loop Closure → Globally Optimized Trajectory**

---

## Code Structure

The repository keeps the course exercise entry points at the project root (`ex3.py`–`ex7.py`). Larger stages are split into focused modules so that the scientific pipeline, workflow orchestration, analysis, and reporting remain distinguishable.

The diagrams below show the main canonical execution and reading flow rather than an exhaustive Python import tree. Tests, validators, and retained historical or experimental helpers are omitted for clarity.

Detailed reader guides for Exercises 4–7 are available under `docs/`.

### Exercise 3 — Stereo Visual Odometry

```text
ex3.py
└── tools.py
    ├── feature detection and stereo matching
    ├── triangulation
    ├── temporal reciprocal matching
    ├── PnP pose estimation
    └── RANSAC / four-view supporter classification
```

`ex3.py` presents the reader-facing Visual Odometry stages.

`tools.py` contains the shared image-processing and geometric primitives used for feature extraction and matching, stereo triangulation, PnP pose estimation, RANSAC, and trajectory construction. These primitives are also reused by later Tracking and Loop Closure stages.

### Exercise 4 — Tracking Database

```text
ex4.py
└── exercise4_workflow.py
    ├── exercise4_database_builder.py
    │   ├── tracking_database_adapter.py
    │   ├── tracking_database.py
    │   └── tools.py
    └── exercise4_analysis.py
```

`ex4.py` is the command-line entry point for the Tracking Database stage.

`exercise4_database_builder.py` contains the sequential scientific pipeline: stereo extraction, temporal matching, four-view consensus, PnP/RANSAC pose estimation, and insertion of accepted correspondences into the Tracking Database.

`tracking_database.py` defines the database itself, while `tracking_database_adapter.py` connects the Exercise 3 correspondence representation to the database row/track representation.

`exercise4_workflow.py` owns the surrounding build, serialization, reload, and analysis workflow.

`exercise4_analysis.py` consumes the completed Tracking Database and produces tracking statistics, representative-track analysis, reprojection diagnostics, and report figures.

### Exercise 5 — Sliding-Window Bundle Adjustment

```text
ex5.py
├── exercise5_full_run.py
├── exercise5_bundle_window.py
├── exercise5_geometry.py
├── exercise5_artifacts.py
├── exercise5_aggregation.py
├── exercise5_reporting.py
└── exercise5_submission.py
```

`ex5.py` is the explicit command-line boundary for the Bundle Adjustment workflow and delegates the scientific computation to the underlying modules.

`exercise5_full_run.py` orchestrates the ordered sequence of local Bundle windows and extracts the endpoint measurements and covariances used downstream.

`exercise5_bundle_window.py` constructs and solves one local Bundle Adjustment problem using camera poses, tracked 3D landmarks, stereo reprojection factors, and a gauge-fixing prior.

`exercise5_geometry.py` contains the coordinate-system and camera-geometry helpers used by the Bundle solver.

`exercise5_artifacts.py` owns deterministic persistence and sealing of Bundle-window evidence. The aggregation, reporting, and submission modules operate on the retained results without changing the scientific Bundle solution.

The compact output passed to Exercise 6 is the optimized relative pose between each pair of consecutive keyframes together with its conditional covariance.

`exercise5_track_survival.py` is retained only for a completed alternative keyframe-policy experiment and is not part of the selected canonical pipeline.

### Exercise 6 — Pose Graph

```text
ex6.py
├── PoseGraph.py
└── exercise6_reporting.py
```

`ex6.py` is the thin runner for the global Pose Graph stage.

`PoseGraph.py` loads the ordered relative-pose measurements and conditional covariances produced by Exercise 5, constructs the keyframe graph, adds the consecutive `BetweenFactorPose3` constraints, builds the composed initial trajectory, and performs the graph optimization.

`exercise6_reporting.py` generates the Pose Graph outputs, trajectory comparisons, covariance visualizations, and report-ready artifacts.

### Exercise 7 — Loop Closure

```text
ex7.py
├── exercise7_sequential_loop_closure.py
│   ├── exercise7_candidate_detection.py
│   ├── exercise7_consensus_matching.py
│   ├── exercise7_small_bundle.py
│   └── exercise7_pose_graph_update.py
└── exercise7_report_freeze.py
```

`ex7.py` is the command-line entry point for preflight checks, testing, validation, reporting, submission materialization, and the explicitly confirmed canonical Loop Closure run.

`exercise7_sequential_loop_closure.py` orchestrates the complete sequential Loop Closure process.

`exercise7_candidate_detection.py` implements the course candidate-detection stage: relative covariance is approximated along a covariance-graph path and historical keyframes are scored using a Mahalanobis-distance test.

`exercise7_consensus_matching.py` geometrically verifies candidate image pairs using stereo/temporal correspondence, PnP, RANSAC, and four-view supporter tests.

`exercise7_small_bundle.py` solves a small two-pose stereo Bundle Adjustment for a successful loop candidate and extracts the relative-pose measurement and conditional covariance.

`exercise7_pose_graph_update.py` adds the resulting Loop Closure `BetweenFactorPose3` to the cumulative Pose Graph, re-optimizes it, and recomputes the pose Marginals before the update is accepted.

`exercise7_report_freeze.py` is deliberately separate from the scientific pipeline: it reads validated sealed results and generates report/submission material without rerunning matching, Bundle Adjustment, Pose Graph optimization, or Marginals.

---

## Project Roadmap

The system was developed incrementally, with each course exercise introducing the next layer of the SLAM pipeline.

### Foundations — Exercises 1–2

The first exercises introduced the basic stereo-vision building blocks used by the later pipeline, including feature correspondence, rectified-stereo geometric filtering, and 3D triangulation.

The original Exercise 1 and Exercise 2 implementations depended on a legacy helper module that was removed during the final repository cleanup. The current `ex1.py` and `ex2.py` files are therefore historical placeholders, while the final runnable SLAM pipeline begins with the validated Exercise 3 implementation.

### Exercise 3 — Stereo Visual Odometry

Consecutive stereo pairs are connected through feature matching. Points triangulated from the previous stereo frame provide 3D-to-2D correspondences for PnP pose estimation.

A custom RANSAC procedure evaluates candidate poses using reprojection consistency across all four images of two consecutive stereo pairs. The accepted relative poses are accumulated to obtain the initial vehicle trajectory.

### Exercise 4 — Multi-Frame Tracking Database

The frame-to-frame correspondences are converted into persistent feature tracks spanning multiple frames.

The Tracking Database associates stereo observations with `TrackId`s and `FrameId`s and stores the measurements and frame-pose information required by the later optimization stages.

### Exercise 5 — Sliding-Window Bundle Adjustment

The trajectory is divided into small windows bounded by keyframes.

For each window, camera poses and tracked 3D landmarks are jointly optimized using GTSAM stereo reprojection factors. Each Bundle is expressed in the coordinate system of its first keyframe, keeping the optimization local.

The optimized relative transformation between the endpoint keyframes, together with its conditional covariance, becomes the measurement used by the Pose Graph.

### Exercise 6 — Global Pose Graph

The local Bundle results are reduced to a compact graph containing only keyframe poses and relative pose constraints between consecutive keyframes.

Each Bundle endpoint measurement is inserted as a GTSAM `BetweenFactorPose3` with its associated covariance. The graph is initialized by composing the consecutive relative motions, optimized globally, and followed by marginal-covariance computation for all keyframe poses.

Because the Pose Graph contains keyframe poses and pose-to-pose constraints rather than the landmarks and reprojection factors of the Bundle problems, it provides a concise representation of the full trajectory.

### Exercise 7 — Loop Closure

The Pose Graph is extended with constraints between non-consecutive keyframes when the vehicle revisits a previously observed location.

For each current keyframe, the system searches previous keyframes using a relative-covariance approximation and a Mahalanobis-distance candidate test. Promising candidates are then verified by four-view consensus matching and PnP/RANSAC.

For a successful match, a small two-pose Bundle Adjustment estimates the relative pose and conditional covariance between the two keyframes. The resulting Loop Closure factor is added to the cumulative Pose Graph, which is immediately re-optimized before processing continues.

On the canonical KITTI Sequence 00 run, the system accepted **6 Loop Closure constraints**.

---

## Canonical Run

The final canonical pipeline uses:

- **3,360** stereo frame pairs from KITTI Odometry Sequence 00
- **356** selected keyframes
- **355** local Bundle Adjustment windows
- a **356-node** consecutive-keyframe Pose Graph before Loop Closure
- **6** accepted Loop Closure constraints

---

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV, SIFT, stereo matching, triangulation, PnP, RANSAC
- **Optimization:** GTSAM, stereo reprojection factors, Bundle Adjustment, Pose Graph optimization
- **Numerical Computing:** NumPy
- **Dataset:** KITTI Odometry Benchmark — Sequence 00

---


### Generic SLAM Pipeline

`generic_slam/` is a sequence-independent refactor of the original SLAM pipeline, designed to run the same stages on different KITTI odometry sequences. It was additionally tested on KITTI sequence 05.
