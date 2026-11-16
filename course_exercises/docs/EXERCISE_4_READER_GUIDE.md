# Exercise 4 Reader Guide

## 1. Exercise 4 purpose

Exercise 4 turns the consecutive-frame stereo visual-odometry machinery from
Exercise 3 into a persistent tracking database for KITTI sequence 00. For each
frame, it stores stereo observations as `Link` rows; across consecutive frames,
it uses only the final four-view PnP/RANSAC supporters to create or extend
tracks. It also preserves the estimated world-to-camera poses and per-transition
supporter evidence outside the lecturer `TrackingDB`.

The completed database is then reloaded and analyzed for track lengths, links
per frame, consecutive-frame connectivity, supporter percentages, trajectory
quality, representative tracks, and stereo reprojection error.

```text
Exercise 3 stereo and temporal geometry
        ↓
Exercise 4 sequential TrackingDB construction
        ↓
Serialized TrackingDB + companion pose/supporter metadata
        ↓
Exercise 4 tracking, trajectory, and reprojection analysis
        ↓
Exercise 5 Bundle Adjustment input
```

## 2. Recommended reading order

1. `ex4.py` — start here for the `build`, `analyze`, and `all` command flow.
2. `exercise4_database_builder.py` — follow the scientific sequential build,
   especially `build_exercise4_database` and
   `_estimate_consecutive_frame_transition`.
3. `tracking_database_adapter.py` — understand how raw Exercise 3 match indices
   become row-complete inputs for the lecturer database.
4. `tracking_database.py` — inspect the protected lecturer `Link` and
   `TrackingDB` definitions, especially `TrackingDB.add_frame`.
5. `exercise4_workflow.py` — see output safety, serialization, companion
   materialization, and the build/analysis boundary.
6. `exercise4_analysis.py` — follow verified reload, statistics,
   representative-track output, reprojection, and figures.
7. `tools.py` — consult the shared Exercise 3 primitives named in the stage map
   below; it is a dependency rather than the Exercise 4 entry point.

## 3. Algorithm and pipeline map

```text
KITTI left/right stereo images
    ↓
SIFT feature extraction, stereo matching, and stereo-inlier filtering
    ↓
TrackingDB row preparation for each stereo-valid left feature
    ↓
Back-frame stereo triangulation using its world-to-camera pose
    ↓
Reciprocal matching between consecutive left images
    ↓
Four-view consensus: back-left/right + front-left/right
    ↓
PnP inside the custom RANSAC loop; refine and retain final supporters
    ↓
Map raw temporal indices into previous/current TrackingDB row indices
    ↓
Insert a row-complete transition whose validity mask selects supporters only
    ↓
TrackingDB.add_frame creates new tracks, extends existing tracks, and leaves
nonconnected observations untracked
    ↓
Serialize TrackingDB, transition evidence, and companion pose/supporter state
    ↓
Reload and verify the serialized database/companion pair
    ↓
Track, frame-link, connectivity, supporter, and trajectory statistics
    ↓
Representative-track output, reprojection analysis, and required figures
```

The pose convention retained in the companion is world-to-camera,
`x_i = R_i X + t_i`. A stereo `Link` stores `(x_left, x_right, y)`, with the
shared vertical coordinate formed from the two stereo observations.

## 4. Exact stage-to-code map

| Course / algorithm stage | File | Function / class | Responsibility |
| --- | --- | --- | --- |
| Exercise 4 CLI and dispatch | `ex4.py` | `main`, `build_parser`, `_run_build`, `_run_analyze`, `_run_all` | Validates command arguments and dispatches build and analysis. `all` runs them in separate processes. |
| Build/analysis workflow boundary | `exercise4_workflow.py` | `build_and_serialize_tracking_database`, `analyze_serialized_tracking_database` | Wraps the scientific builder with safe output creation, artifact materialization, and serialized analysis. |
| Sequential database build | `exercise4_database_builder.py` | `build_exercise4_database` | Processes frames in order, initializes frame 0, estimates every consecutive transition, inserts supporters, and returns the database plus evidence. |
| Canonical stereo-frame extraction | `exercise4_database_builder.py` | `extract_canonical_stereo_frame` | Loads a KITTI stereo pair, invokes the shared feature/stereo filter, and prepares both raw and TrackingDB-row representations. |
| Image loading | `tools.py` | `read_images_from_dataset` | Loads the left and right KITTI images for one FrameId. |
| Stereo feature extraction and filtering | `tools.py` | `extract_keypoints_and_inliers` | Detects SIFT features, matches left to right, and keeps stereo inliers satisfying vertical agreement and positive disparity. |
| Stereo row preparation | `tracking_database_adapter.py` | `prepare_stereo_frame`, `PreparedStereoFrame` | Deduplicates and orders stereo-valid left features, calls `TrackingDB.create_links`, and records raw-to-filtered row mappings. |
| Back-frame triangulation | `exercise4_database_builder.py` | `_estimate_consecutive_frame_transition` | Forms the back-right pose and delegates triangulation before estimating the next pose. |
| Stereo triangulation primitive | `tools.py` | `calculate_right_camera_matrix`, `cv_triangulate_matched_points` | Forms the right-camera extrinsics and triangulates the back stereo inliers. |
| Reciprocal temporal matching | `tools.py` | `match_reciprocal` | Matches consecutive left-image descriptors in both directions and retains reciprocal matches. |
| Four-view consensus | `tools.py` | `find_consensus_matches_indices` | Retains temporal matches whose endpoints are stereo inliers in both frames; returns back-stereo, front-stereo, and temporal indices. |
| PnP and RANSAC pose estimation | `tools.py` | `estimate_projection_matrices_with_ransac`, `build_model`, `calculate_front_camera_matrix` | Samples consensus correspondences, solves front-left PnP, scores four-camera reprojections, refines the best model, and returns final supporters. |
| Consecutive transition result | `exercise4_database_builder.py` | `_estimate_consecutive_frame_transition`, `_EstimatedConsecutiveTransition` | Collects reciprocal matches, consensus, final supporters, the front world-to-camera pose, and model counters. |
| Final supporter handling | `exercise4_database_builder.py` | `_insert_supported_transition` | Sends final supporters to the adapter, checks supporter-only insertion, and records track lifecycle evidence. |
| Raw-index to TrackingDB-row adaptation | `tracking_database_adapter.py` | `prepare_temporal_transition`, `PreparedTemporalTransition` | Converts raw match indices to filtered row indices, resolves duplicate row/destination candidates deterministically, and builds the complete previous-row match list and supporter validity mask required by `TrackingDB.add_frame`. |
| Frame/transition insertion boundary | `tracking_database_adapter.py` | `Exercise3TrackingDBAdapter.add_initial_frame`, `Exercise3TrackingDBAdapter.add_frame` | Validates sequential state, delegates mutation to `TrackingDB.add_frame`, and appends companion metadata. |
| Track creation and extension | `tracking_database.py` | `TrackingDB.add_frame` | Creates a track when a valid match first connects two frames, extends an existing track on later valid matches, and records leftover untracked links. |
| Database definition | `tracking_database.py` | `Link`, `TrackingDB` | Defines stereo observations, frame/track indices, mutation, queries, and pickle serialization. This lecturer file is protected. |
| Companion pose/supporter metadata | `tracking_database_adapter.py` | `TrackingDBCompanion` | Stores FrameId-aligned world-to-camera poses and transition-aligned supporter counts, percentages, and frame pairs outside `TrackingDB`. |
| Transition evidence | `exercise4_database_builder.py` | `_build_transition_metadata` | Records fixed-order feature, consensus, supporter, PnP/RANSAC, connectivity, and track-lifecycle evidence for each consecutive pair. |
| Serialization and materialization | `exercise4_workflow.py` | `build_and_serialize_tracking_database` | Calls `TrackingDB.serialize` and writes the production companion, transitions table, build summary, hashes, and configuration evidence. |
| Verified artifact reload | `exercise4_analysis.py` | `load_tracking_artifacts` | Verifies file identity and schema, loads the pickle, validates database consistency and its streaming digest, and checks database/companion alignment. |
| Tracking and trajectory statistics | `exercise4_analysis.py` | `compute_database_metrics`, `_compute_track_statistics`, `_compute_frame_and_transition_statistics`, `_trajectory_metrics` | Computes track-length, per-frame link, connectivity, supporter, lifecycle, and trajectory summaries and writes metric tables. |
| Representative-track selection/output | `exercise4_analysis.py` | `_compute_track_statistics`, `write_representative_track` | Selects a deterministic eligible track and writes its observations and image/crop figure. |
| Reprojection analysis | `exercise4_analysis.py` | `compute_reprojection` | Triangulates the selected track from its first and last stereo observations using ground-truth poses, then measures left/right reprojection error over the track. |
| Figure generation | `exercise4_analysis.py` | `write_figures` | Writes track-length, links-per-frame, connectivity, supporter-percentage, and reprojection figures. |
| Top-level analysis orchestration | `exercise4_analysis.py` | `analyze_tracking_database` | Runs verified reload, metrics, representative-track output, reprojection, figures, and final deterministic artifact metadata in order. |

The central mutation call path is:

```text
exercise4_database_builder._insert_supported_transition
    → Exercise3TrackingDBAdapter.add_frame
        → prepare_temporal_transition
        → TrackingDB.add_frame
```

This boundary matters because Exercise 3 matches use raw keypoint/descriptor
indices, while the lecturer `TrackingDB` accepts rows filtered to stereo-valid
features and requires one ordered temporal entry per previous-frame row.

## 5. File responsibilities

- `ex4.py` — thin CLI, argument validation, and process-isolated dispatch.
- `exercise4_database_builder.py` — scientific sequential TrackingDB
  construction and transition evidence.
- `tracking_database_adapter.py` — representation conversion and the validated
  mutation boundary between Exercise 3 correspondences and lecturer row/track
  semantics.
- `tracking_database.py` — protected lecturer data structure and serialization
  implementation.
- `exercise4_workflow.py` — output safety, serialization/materialization,
  provenance packaging, and build/analyze workflow.
- `exercise4_analysis.py` — deterministic analysis of an already-built,
  independently reloaded database; it performs no feature extraction or motion
  estimation.
- `tools.py` — protected shared Exercise 3 feature, matching, triangulation,
  PnP, and RANSAC primitives reused by the builder.

The first three scientific stages live primarily in the builder and shared
primitives. Workflow and analysis code deliberately do not duplicate them.

## 6. Build versus analysis

Exercise 4 separates the expensive scientific build from read-only analysis:

```text
Build:    images → stereo/temporal geometry → TrackingDB → serialized artifacts
Analysis: serialized artifacts → verified reload → metrics/tracks/reprojection/figures
```

`exercise4_workflow.py` enforces this boundary and owns safe output-directory
handling. `ex4.py` remains thin so the `all` command can finish the build
process before starting analysis in a fresh process. The analysis therefore
consumes the serialized interface rather than an in-memory builder result.

## 7. Exercise 5 handoff

Exercise 5 consumes the canonical `tracking_database.pkl` and
`tracking_database_companion.json` through
`exercise4_analysis.load_tracking_artifacts`. The database supplies tracks,
FrameIds, stereo `Link` observations, and descriptors; the companion supplies
the FrameId-aligned world-to-camera pose sequence and verified artifact
identity. Exercise 5 adapts these inputs without modifying the lecturer
`TrackingDB`.

## 8. Running Exercise 4

Run commands from the repository root. The generic command forms are:

```bash
python ex4.py build --frame-count <FRAME_COUNT> --output-dir <OUTPUT_DIR>

python ex4.py analyze \
    --database <TRACKING_DATABASE.pkl> \
    --companion <TRACKING_DATABASE_COMPANION.json> \
    --output-dir <OUTPUT_DIR>

python ex4.py all --frame-count <FRAME_COUNT> --output-dir <OUTPUT_DIR>
```

To analyze the existing canonical Exercise 4 artifacts without rebuilding the
database, write the new analysis to a separate output root:

```bash
python ex4.py analyze \
    --database diagnostics/ex4_final/tracking_database.pkl \
    --companion diagnostics/ex4_final/tracking_database_companion.json \
    --output-dir diagnostics/ex4_manual_analysis
```

A fresh full 3360-frame database build can be written to a new output root:

```bash
python ex4.py build \
    --frame-count 3360 \
    --output-dir diagnostics/ex4_manual_full_run
```

To run the full build followed by analysis in separate processes:

```bash
python ex4.py all \
    --frame-count 3360 \
    --output-dir diagnostics/ex4_manual_all
```

`diagnostics/ex4_final` is protected canonical evidence. Never use it, or a
directory inside it, as a new output directory. Output roots must be new or
empty according to the selected command's safety checks.

A 3360-frame build is expensive. Readers who only want to inspect and
regenerate analysis for the existing result should use `analyze` with the
canonical database and companion rather than rebuilding the full sequence.
