# Exercise 5 Reader Guide

## 1. Exercise 5 purpose

Exercise 5 turns the frame-by-frame tracking result from Exercise 4 into a
sequence of locally optimized Bundle Adjustment problems. Exercise 4 supplies
two immutable inputs: the canonical `TrackingDB`, which contains the stereo
tracks and measurements, and its companion evidence, which contains the
world-to-camera PnP trajectory. Exercise 5 uses that trajectory to initialize
camera poses and the database tracks to construct stereo reprojection factors.

The course requirements are divided across four related tasks:

- Section 5.1 examines one sufficiently long track and relates direct stereo
  reprojection error to the corresponding GTSAM factor error.
- Section 5.2 establishes the camera-pose convention needed to express every
  pose and landmark in a Bundle window relative to its first frame.
- Section 5.3 constructs and optimizes one well-formed local Bundle, including
  its camera poses, landmarks, stereo factors, gauge prior, error statistics,
  and first-window visual evidence.
- Section 5.4 repeats that computation over the full frozen keyframe sequence,
  composes a Bundle trajectory, and compares it with PnP and ground truth.

Keyframes bound the local windows. Each window is expressed in its own first
frame's coordinate system, so it remains numerically local and has an explicit
gauge anchor. The compact output needed downstream is not the entire Bundle
graph. It is the optimized relative pose between each pair of consecutive
keyframes together with its conditional covariance. Exercise 6 consumes those
355 pose/covariance edges to form a 356-node Pose Graph.

This guide describes the current canonical implementation. It intentionally gives
file and function names but no final source line numbers; final Exercise 4--7
file/function/line traceability is deferred until the whole source layout is
frozen.

## 2. Recommended reading order

For the canonical path, read the files in this order:

1. `ex5.py` -- the thin command-line dispatcher and its safety gates.
2. `exercise5_production_config.json` -- the frozen inputs, keyframes,
   scientific policy, schemas, and output contract.
3. `exercise5_full_run.py` -- configuration validation, one-database-load
   streaming, covariance extraction, and full-run validation.
4. `exercise5_bundle_window.py` -- construction and optimization of one local
   Bundle graph.
5. `exercise5_geometry.py` -- the coordinate-convention conversions used by
   the window solver.
6. `exercise5_artifacts.py` -- deterministic per-window serialization,
   validation, atomic sealing, run state, manifests, and checksums.
7. `exercise5_aggregation.py` -- machine-readable full-population tables built
   from sealed windows.
8. `exercise5_reporting.py` -- figures built from window and aggregate
   evidence.
9. `exercise5_submission.py` -- fail-closed materialization of the curated
   Bundle submission package.

`exercise5_track_survival.py` and `exercise5_track_survival_082_config.json`
document a completed but rejected keyframe-policy
experiment. They are not part of the selected production path. The older
`BundleWindow.py` and `BundleAdjustment.py` files are legacy course code; they
are neither the canonical implementation nor the path invoked by `ex5.py`.

## 3. Algorithm and pipeline map

```text
canonical E4 TrackingDB + companion PnP poses
    -> validate frozen configuration and canonical input identity
    -> read the frozen ordered keyframe population
    -> select one consecutive [start, end] window
    -> express cameras and landmarks in the start-frame coordinate system
    -> build one anchored stereo Bundle graph
    -> evaluate initial graph and factor errors
    -> optimize with GTSAM Levenberg-Marquardt
    -> construct GTSAM Marginals
    -> extract optimized endpoint poses
    -> extract ordered [start, end] joint covariance/information
    -> compute the end-given-start conditional covariance
    -> store start_pose.between(end_pose)
    -> prepare, validate, and atomically seal the window artifacts
    -> release the graph-bearing result and Marginals
    -> continue with the next consecutive window
    -> validate all sealed windows
    -> aggregate trajectories and error tables
    -> generate report figures
    -> materialize a fail-closed curated submission package
    -> hand relative-pose/covariance edges to Exercise 6
```

The streaming boundary is deliberate: the database is loaded once, but a
Bundle graph and its Marginals exist for only one window at a time.

## 4. Exact stage -> file -> function/class map

| Course/algorithm stage | File | Function/class | Responsibility |
|---|---|---|---|
| E4 canonical input loading | `exercise4_analysis.py`; `exercise5_full_run.py` | `load_tracking_artifacts`; `_load_canonical_once` | Load and validate the lecturer `TrackingDB`, companion poses/transitions, schemas, hashes, counts, and streaming digest exactly once for a production invocation. |
| Frozen configuration loading | `exercise5_full_run.py` | `load_frozen_configuration`; `load_run_configuration` | Read and validate the active JSON configuration for the selected production path. |
| Configuration and canonical-input validation | `exercise5_full_run.py` | `validate_frozen_configuration`; `validate_source_and_input_hashes` | Check required policy/schema fields and canonical E4 input identities before deserializing scientific inputs. |
| Frozen ordered keyframe population | `exercise5_production_config.json`; `exercise5_full_run.py` | `keyframe_selection.keyframes`; `run_streamed` | Supply and consume the fixed 356-element ordered list. Production does not rediscover keyframes. |
| Consecutive window endpoint selection | `exercise5_full_run.py` | `run_streamed` | Form window `i` from `keyframes[i]` and `keyframes[i + 1]`, giving exactly 355 windows. |
| First-frame-local camera transform | `exercise5_geometry.py` | `camera_pose_in_window` | Convert canonical world-to-camera extrinsics into `T_Cstart_from_Ci`, with the first pose equal to identity. |
| Camera keys and initial `Pose3` values | `exercise5_bundle_window.py` | `solve_bundle_window` | Create actual-frame-ID `c` keys and insert each initial camera pose in deterministic frame order. |
| Candidate track collection | `exercise5_bundle_window.py` | `_ordered_frame_tracks` | Collect the sorted tracks seen in each inclusive frame of the window and form the candidate population. |
| Observation validation | `exercise5_bundle_window.py` | `_collect_valid_observations` | Validate membership, finite stereo coordinates, and positive disparity, and retain valid measurements with explicit rejection evidence. |
| Landmark eligibility policy | `exercise5_bundle_window.py` | `ProbeLandmarkEligibility`; `solve_bundle_window` | Apply the caller-supplied minimum observations and positive/depth requirements without deriving a separate production policy inside the solver. |
| Landmark initialization | `exercise5_bundle_window.py` | `_initialize_landmark` | Try valid observations from latest to earliest, stereo-backproject with `gtsam.StereoCamera`, and express the point in the window frame; this is a stereo backprojection path, not a separate monocular triangulator. |
| Bundle graph creation | `exercise5_bundle_window.py` | `solve_bundle_window` | Create one `gtsam.NonlinearFactorGraph`, its initial `gtsam.Values`, and deterministic camera/landmark key maps. |
| First-camera gauge anchor | `exercise5_bundle_window.py` | `solve_bundle_window` | Add exactly one identity `gtsam.PriorFactorPose3` on the first camera with the frozen caller-provided prior noise. |
| Stereo projection factors | `exercise5_bundle_window.py` | `solve_bundle_window`; `ProjectionFactorRecord` | Insert one `gtsam.GenericStereoFactor3D` for each retained valid observation and retain a record connecting the factor to its frame, track, keys, and measurement. |
| Graph connectivity and well-formedness | `exercise5_bundle_window.py` | `_assert_connected`; `_assert_values_finite`; `solve_bundle_window` | Check the camera-landmark bipartite graph, finite initial values, factor/key uniqueness and counts, and that residual dimension is sufficient for the inserted values. |
| Initial graph and factor errors | `exercise5_bundle_window.py` | `_error_metrics`; `_maximum_projection_diagnostic` | Compute raw total/mean graph error, stereo projection-factor statistics, and the maximum projection-factor diagnostic at the initial values. |
| Optimizer construction and solve | `exercise5_bundle_window.py` | `solve_bundle_window` | Construct the default `gtsam.LevenbergMarquardtOptimizer`, optimize the graph, and reject nonfinite output. |
| Optimized camera and landmark values | `exercise5_bundle_window.py` | `BundleWindowResult`; `solve_bundle_window` | Return the optimized `final_values` together with the graph, initial values, maps, eligibility evidence, and initial/final metrics. |
| GTSAM Marginals | `exercise5_full_run.py` | `run_streamed` | Construct `gtsam.Marginals(result.graph, result.final_values)` after a successful window solve. |
| Optimized start/end pose extraction | `exercise5_full_run.py` | `extract_endpoint_edge`; `extract_endpoint_covariances` | Read the optimized first and last camera `Pose3` values using the solver's frame-to-key map. |
| Ordered joint endpoint covariance/information | `exercise5_full_run.py` | `extract_endpoint_covariances` | Request joint Marginals with an explicit `gtsam.KeyVector([start_key, end_key])` and retain the raw 12x12 covariance and information matrices. |
| Conditional covariance validation | `exercise5_full_run.py` | `validate_covariance_evidence` | Validate the joint blocks and compute/validate the 6x6 conditional end-given-start covariance without repairing the scientific evidence. |
| Relative endpoint transform | `exercise5_full_run.py` | `extract_endpoint_edge` | Compute `start_pose.between(end_pose)` and label the stored direction `T_Cstart_from_Cend`. |
| Window artifact preparation and sealing | `exercise5_full_run.py`; `exercise5_artifacts.py` | `seal_window_artifacts`; `_seal_window_artifacts` | Prepare normalized JSON/CSV/NPZ/PNG evidence in an attempt directory, validate it, write hashes last, and atomically promote the sealed window. |
| Prepared/sealed window validation and reload | `exercise5_artifacts.py` | `validate_prepared_window`; `validate_sealed_window` | Enforce schema, shape, ordering, required-file, and checksum contracts both before sealing and when reloading or resuming. |
| One-window-at-a-time release | `exercise5_full_run.py` | `run_streamed` | Delete the live `Marginals`, graph-bearing `BundleWindowResult`, covariance evidence, and endpoint data after the window is sealed, then collect before continuing. |
| Full-run validation | `exercise5_full_run.py` | `validate_run_root` | Reload the configuration and every expected sealed window, enforce exact ordered coverage, and validate complete or explicitly allowed partial state. |
| Run state, manifest, and checksums | `exercise5_artifacts.py` | `snapshot_configuration`; `update_run_state`; `write_generator_registry`; `build_run_manifest`; `write_and_seal_manifest`; `write_checksum_index` | Own the deterministic run-level lifecycle records separately from Bundle science. |
| Keyframe trajectory composition | `exercise5_aggregation.py` | `_trajectory_evidence`; `compose_keyframe_trajectories` | Compose saved endpoint edges in order and write the Bundle, PnP, and ground-truth keyframe trajectories. |
| Aggregate error tables | `exercise5_aggregation.py` | `aggregate_window_metrics`; `aggregate_window_rejections`; `aggregate_projection_residuals`; `compute_keyframe_localization_errors`; `compute_consecutive_relative_errors`; `compute_relative_subsections`; `write_low_connectivity_table`; `aggregate_all` | Read the complete sealed population and produce deterministic machine-readable metrics and comparisons. |
| Report figures | `exercise5_reporting.py` | `generate_report_figures` and its `plot_*` helpers | Read existing per-window and aggregate evidence and create the report figure groups, including the three Bundle-versus-PnP comparison figures. |
| Curated submission | `exercise5_submission.py` | `materialize_bundle_package` | Stage, validate, hash, and atomically materialize a new curated package only from a complete canonical source. |
| E6-facing consumer boundary | `PoseGraph.py` | `load_canonical_pose_graph_inputs`; `PoseGraph.create_factor_graph` | Reload the stored consecutive relative poses and conditional covariances, then use them directly in `BetweenFactorPose3` factors. |

## 5. File responsibilities

`exercise5_bundle_window.py` owns the scientific construction and optimization
of one Bundle window. It knows how to validate observations, initialize camera
and landmark values, add the prior and stereo factors, test graph structure,
calculate errors, and return optimized values. It does not choose the
production keyframes, load the canonical database, manage a 355-window run, or
seal production artifacts.

`exercise5_geometry.py` owns small, explicit conversions among canonical
world-to-camera extrinsics, GTSAM camera poses, window-local camera poses, and
points in world/camera/window coordinates. Keeping these conversions separate
makes the direction of every transform testable.

`exercise5_full_run.py` owns the scientific runtime around the window solver:
frozen configuration validation, exact canonical-input checks, one-time
E4 loading, the streaming loop, memory admission, construction of Marginals,
endpoint pose and covariance extraction, resumability, and full-run validation
wrappers. It supplies the frozen noise and landmark policy to the general
single-window solver.

`exercise5_artifacts.py` owns deterministic serialization and lifecycle state.
It prepares and validates a window, seals it atomically, validates a sealed
window on reload, quarantines failed attempts, and writes run configuration,
state, generator, manifest, and checksum evidence. It does not decide the
Bundle geometry or optimizer behavior.

`exercise5_aggregation.py` reads sealed windows and creates aggregate
machine-readable tables. `exercise5_reporting.py` reads window and finalized
aggregate evidence and creates figures. `exercise5_submission.py` performs a
separate fail-closed transaction that materializes a curated package into a
new destination. These are post-window stages, not alternate Bundle solvers.

`ex5.py` is intentionally thin. It parses explicit commands, dispatches to the
modules above, prints a JSON result, and applies confirmation/output gates. It
contains no second implementation of Bundle Adjustment.

`ProbeLandmarkEligibility` is the caller-supplied landmark-policy container
used by both bounded validation and the canonical runner. Despite its
historical name, it supplies no probe-only defaults: production values come
from the frozen configuration, and the solver contains no separate probe
algorithm.

## 6. One Bundle window explained

The canonical companion extrinsics are world-to-camera transforms:

```text
E_i = T_Ci_from_W
```

For a window whose first frame is `a`, the GTSAM camera pose for frame `i` is
expressed relative to camera `a`:

```text
X_i^(a) = E_a E_i^-1 = T_Ca_from_Ci
X_a^(a) = Identity
```

Thus a pose maps coordinates from camera `i` into the window's parent frame
`C_a`. Landmarks initialized from stereo measurements are transformed into
that same window frame. This is the concrete implementation of the Exercise
5 requirement that all values in one Bundle use its first frame as the common
coordinate system.

`solve_bundle_window` performs one solve in the following scientific order:

1. It validates the inclusive `[start, end]` range, the companion extrinsics,
   stereo calibration, noise-model dimensions, and eligibility policy.
2. It constructs actual-frame-ID camera keys and inserts every window-local
   camera `Pose3`; the first pose must be identity.
3. It creates the nonlinear factor graph and adds exactly one identity prior
   on the first camera using the frozen caller-provided prior noise.
4. It collects sorted candidate tracks and validates their stereo observations.
   Invalid membership, nonfinite coordinates, or nonpositive disparity is
   retained as explicit rejection evidence.
5. For each eligible candidate it tries valid observations from latest to
   earliest. `gtsam.StereoCamera.backproject` initializes the landmark, after
   which the point must satisfy the configured finite, positive-depth, and
   initialization-depth rules in the relevant cameras.
6. It inserts accepted landmarks in sorted track order and adds one
   `GenericStereoFactor3D` per retained observation.
7. It verifies keys, counts, finite values, connectedness of the
   camera-landmark bipartite graph, and the graph's residual/value dimensions.
8. It evaluates raw initial graph and per-factor errors and records the maximum
   initial stereo projection-factor diagnostic.
9. It constructs the default GTSAM `LevenbergMarquardtOptimizer`, optimizes,
   checks the result for finite values, and evaluates final errors.
10. It returns a `BundleWindowResult` containing the graph, initial and final
    `Values`, key maps, factor records, eligibility/rejection evidence, and
    initial/final metrics. The streaming caller then constructs Marginals and
    extracts the compact endpoint evidence.

No production keyframe choice, dynamic policy derivation, artifact rewrite, or
Pose Graph construction occurs inside this function.

## 7. Full streamed 355-window workflow

Production reads the ordered `keyframe_selection.keyframes` array from
`exercise5_production_config.json`; it does not derive keyframes dynamically
from the database. The frozen population contains:

- 356 keyframes;
- 355 consecutive windows;
- coverage from frame 0 through frame 3359; and
- between 5 and 20 actual inclusive frames in every window.

`run_streamed` first validates the frozen configuration, canonical input
identities, and the memory gate. It then loads the canonical E4 database and
companion once.
For window index `i`, it uses `keyframes[i]` and `keyframes[i + 1]`, solves the
Bundle, constructs Marginals, extracts and validates endpoint evidence, and
seals the window. Every window serializes its optimized camera and landmark
values; window 0 additionally retains the individual camera marginal
covariances and imagery needed by the first-window report.

The memory policy is part of the production contract:

```text
one TrackingDB load
+ at most one live Bundle graph/result
+ at most one live Marginals object
+ compact sealed evidence from earlier windows
```

After a window is sealed, the live Marginals and graph-bearing result are
released before the next solve. Aggregation later reloads the compact sealed
artifacts; it does not require all Bundle graphs to coexist.

Resumability is seal-based. If a target window directory already exists,
`run_streamed` validates its identity, required files, schemas, and hashes
before skipping it. An incomplete attempt is not treated as a completed
window. New evidence is prepared in an attempt directory, validated, supplied
with its artifact hashes last, and atomically promoted. This makes every skip
an independently justified reuse of a sealed window rather than a trust in a
counter or filename alone.

The accepted clean run is already complete under
`diagnostics/ex5_clean_final`. It contains all 355 validated window indices and
is immutable evidence; the production computation does not need to be rerun
for documentation or report reading.

## 8. Relative pose and covariance handoff to Exercise 6

For every solved window `[start, end]`, `extract_endpoint_edge` obtains the
optimized `Pose3` values through `result.camera_key_by_frame_id`. It computes:

```text
relative_pose = start_pose.between(end_pose)
direction     = T_Cstart_from_Cend
```

Because both poses belong to the same start-frame-local Bundle, this is the
optimized transform between the consecutive keyframe cameras in the direction
required by the downstream factor.

`extract_endpoint_covariances` supplies an explicit ordered key vector:

```text
[start_key, end_key]
```

to GTSAM Marginals. It retains both the 12x12 joint covariance and the 12x12
joint information matrix. If the ordered covariance is block-partitioned as

```text
[ Sigma_ss  Sigma_se ]
[ Sigma_es  Sigma_ee ]
```

the stored 6x6 end-given-start conditional covariance is

```text
Sigma_e_given_s = Sigma_ee - Sigma_es solve(Sigma_ss, Sigma_se)
```

`validate_covariance_evidence` checks finiteness, shape, symmetry, positive
definiteness where required, the ordered blocks, and covariance/information
cross-check diagnostics. The raw matrices and any warning-only numerical
cross-check discrepancies remain evidence; the implementation does not
silently replace them.

The `Pose3` tangent coordinate order is rotation x/y/z followed by translation
x/y/z. The bounded E5-I04 contract validation established that the saved
conditional covariance is already in the residual tangent basis used by a
GTSAM `BetweenFactorPose3(start_key, end_key, relative_pose, noise_model)`.
Exercise 6 therefore passes it directly to
`gtsam.noiseModel.Gaussian.Covariance`. No additional Adjoint transport,
scaling, regularization, clipping, or basis change is required.

The current repository integration boundary is intentionally asymmetric. The
clean selected E5 scientific root is `diagnostics/ex5_clean_final`, but
`PoseGraph.py::load_canonical_pose_graph_inputs` still intentionally accepts
the protected historical equivalent root `diagnostics/ex5_final`. Repointing
Exercise 6 to the clean root is a separate clean-integration task. This reader
guide does not modify E6 or imply that the current loader has already been
repointed. The historical and clean E5 roots have byte-identical endpoint
relative-pose and covariance artifacts for all 355 windows.

## 9. Aggregation, reporting, and submission separation

These stages operate after window science has been sealed.

`exercise5_aggregation.py::aggregate_all` begins by calling
`validate_run_root`. Its table producers then write:

- per-window graph, factor, runtime, memory, and connectivity metrics through
  `aggregate_window_metrics`;
- landmark rejection counts through `aggregate_window_rejections`;
- the full projection-residual population through
  `aggregate_projection_residuals`;
- composed Bundle/PnP/ground-truth trajectories through
  `compose_keyframe_trajectories`;
- absolute keyframe errors through `compute_keyframe_localization_errors`;
- consecutive relative translation/rotation errors through
  `compute_consecutive_relative_errors`;
- 100/400/800-frame subsection comparisons through
  `compute_relative_subsections`; and
- low-connectivity evidence through `write_low_connectivity_table`.

`exercise5_reporting.py::generate_report_figures` reads these finalized tables
and the retained first-window/track evidence. Its plotting groups cover window
factor and projection statistics, projection error versus distance from the
Bundle start, Bundle/PnP/ground-truth trajectories, absolute localization
error, consecutive and subsection relative errors, first-window camera
covariances, and the three post-run Bundle-versus-PnP comparison figures:

- `bundle_vs_pnp_absolute_localization_error.png`;
- `bundle_vs_pnp_relative_translation_error.png`; and
- `bundle_vs_pnp_relative_rotation_error.png`.

The reusable helper currently generates 13 aggregate/report figures. The
scientific root `diagnostics/ex5_clean_final` remains sealed with the original
10 figures created by its canonical run. The three later presentation-only
figures were generated without a Bundle rerun and are included in the curated
`project_submission/bundle_clean_final` package.

`exercise5_submission.py::materialize_bundle_package` is a separate,
fail-closed transaction. It accepts only an exact complete canonical source,
validates the source and every window, requires a new destination, copies the
curated data/figures into a staging directory, adds the post-run comparison
figures, builds metrics and a content-hashed manifest, validates the staged
package, and only then promotes it. The immutable scientific root remains the
source of truth; the curated package is a smaller reader-facing selection, not
a replacement scientific run.

## 10. Running Exercise 5

The current parser exposes the commands below. Use the repository's isolated
environment:

```bash
.venv/bin/python ex5.py --help
```

Validation is read-only with respect to a run root:

```bash
.venv/bin/python ex5.py validate --source-root <SEALED_RUN_ROOT>
.venv/bin/python ex5.py validate --source-root <PARTIAL_RUN_ROOT> --allow-partial
```

Aggregation and reporting write under the supplied source root. Use them only
with a disposable or explicitly intended working run root, not with the
protected canonical/historical evidence roots:

```bash
.venv/bin/python ex5.py aggregate --source-root <WORKING_RUN_ROOT>
.venv/bin/python ex5.py aggregate --source-root <WORKING_PARTIAL_ROOT> --allow-partial
.venv/bin/python ex5.py report --source-root <WORKING_RUN_ROOT>
```

Submission materialization requires a complete canonical source and a new
destination:

```bash
.venv/bin/python ex5.py submission \
  --source-root <COMPLETE_WORKING_RUN_ROOT> \
  --destination <NEW_CURATED_DESTINATION>
```

Two scientific-computation commands also exist, but they are not documentation
workflows:

```bash
.venv/bin/python ex5.py probe --output-root /tmp/e5_i03a_<NEW_NAME>
.venv/bin/python ex5.py run \
  --output-root <CANONICAL_RUN_ROOT> \
  --confirm RUN-CANONICAL-355-WINDOWS
```

`probe` performs a real Bundle solve of exactly the frozen first window, must
use a new `/tmp/e5_i03a_*` root, and is not required to read or validate this
documentation. `run` performs or resumes the protected 355-window canonical
computation. The clean canonical run is already complete, so it must not be
run during a documentation pass.

The parser retains three historical experiment workflows:

```bash
.venv/bin/python ex5.py derive-track-survival-082
.venv/bin/python ex5.py run-track-survival-082 --confirm RUN-TRACK-SURVIVAL-082
.venv/bin/python ex5.py finalize-track-survival-082
```

They belong only to the completed rejected 0.82 track-survival experiment and
its protected evidence. They are not alternate recommended production
commands and should not be rerun or used as Exercise 6 input.
