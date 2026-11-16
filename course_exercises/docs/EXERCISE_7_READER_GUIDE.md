# Exercise 7 Reader Guide

## 1. Purpose and pipeline position

Exercise 7 adds Loop Closure to the visual-SLAM trajectory built in the
preceding exercises. For the canonical KITTI sequence 00 run, the dependency
chain is:

```text
E5 Bundle endpoint relative poses and conditional covariances
    -> E6 consecutive-keyframe Pose Graph without Loop Closure
    -> E7 revisit detection and geometric verification
    -> final cumulative Pose Graph with accepted Loop Closure factors
```

E5 supplies 356 ordered keyframes spanning actual FrameIds 0 through 3359 and
355 directed consecutive constraints. E6 builds the no-loop graph from those
constraints. E7 considers every historical/current keyframe pair, proposes
likely revisits, verifies image geometry, constructs a two-camera Bundle
measurement for a passing pair, and updates the cumulative Pose Graph.

In this project, a Loop Closure is recognition of previously observed scene or
location evidence. It is a revisit constraint; it is not evidence that the
camera discovered a new area.

## 2. File map

| File | Responsibility and major entry points | Intentionally does not own |
| --- | --- | --- |
| `ex7.py` | Safe CLI dispatch: `preflight`, guarded `run`, `validate`, `report`, `submission`, and ordered `all` | Candidate mathematics, matching, Bundle construction, or graph optimization |
| `exercise7_candidate_detection.py` | Coherent E5/E6 loading, directed covariance graph, course covariance-path heuristic, and GTSAM Mahalanobis scoring | Temporal policy, image matching, or acceptance |
| `exercise7_consensus_matching.py` | Stereo and reciprocal temporal matching, four-view consensus, bounded PnP/RANSAC, supporter evidence, and the 30/30/30 match gate | Small Bundle measurement or Pose Graph mutation |
| `exercise7_small_bundle.py` | Two-pose stereo Bundle and raw conditional end-given-start covariance extraction | Cumulative graph insertion or global optimization |
| `exercise7_pose_graph_update.py` | Explicit E5/E6 bootstrap and transactional one-factor cumulative Pose Graph update | Candidate selection, matching, or artifact scheduling |
| `exercise7_sequential_loop_closure.py` | Frozen production policy, per-current orchestration, cumulative state, evidence, sealing, replay, and final scientific reports | CLI safety boundary or the later report-freeze package |
| `exercise7_report_freeze.py` | Read-only loading of sealed results, presentation figures, scalar summaries, traceability, and curated submission packaging | Any new matching, RANSAC, Bundle, Pose Graph optimization, or Marginals construction |

## 3. Core conventions

### FrameId and ordinal

An actual `FrameId` is the KITTI image index, such as 63, 1478, or 3359. A
keyframe ordinal is the position in the canonical 356-keyframe list, from 0 to
355. They are not interchangeable:

- GTSAM camera symbols are `gtsam.symbol("c", actual_frame_id)`.
- Pose Graph and covariance-graph nodes are actual FrameIds.
- Temporal gaps, ranking tie-breaks, and suppression rectangles use keyframe
  ordinals.

For example, accepted pair FrameIds `63 -> 1478` occupies ordinals `10 -> 160`.

### Pose and factor direction

For endpoints `previous` and `current`, the measurement is
`previous_pose.between(current_pose)`. Under the validated camera convention,
this is `T_Cprevious_from_Ccurrent`: it maps coordinates expressed in the
current camera into the previous camera frame. It is stored in a
`BetweenFactorPose3(previous_key, current_key, measurement, noise)` and is
described operationally as the previous-to-current factor direction.

Every accepted Loop Closure covariance has the same direction as its pose
measurement: conditional current/end given previous/start.

### Pose3 tangent and units

All 6-vectors and 6x6 pose covariances use the validated GTSAM order:

```text
[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
```

Rotation entries are in radians and translation entries are in metres.
Scientific factors use the raw saved matrices directly. No covariance is
scaled, clipped, regularized, pseudoinverted, transported with an Adjoint, or
changed to another basis.

## 4. Candidate detection

`load_candidate_detection_inputs` accepts an explicit coherent E5/E6 pair.
The clean defaults are `diagnostics/ex5_clean_final` and
`diagnostics/ex6_clean_final`. The loader verifies that E6 names that E5 root
and configuration, that the 356 actual FrameIds agree, and that its optimized
poses reconstruct the 355 E5 consecutive relative poses. It does not load the
TrackingDB, images, or a Bundle graph.

`CovarianceGraph` is directed. Initially it contains one edge per consecutive
keyframe window; successfully integrated Loop Closures add directed shortcut
edges. Each edge carries the raw 6x6 conditional end-given-start covariance in
the common Pose3 tangent contract.

The lecturer/course path heuristic associates a full matrix label with each
visited node. Along a proposed path `p`, it directly accumulates

```text
Sigma_p = sum of the raw edge covariance matrices on p.
```

The priority and relaxation comparison use `det(Sigma_p)`, implemented with
`slogdet` for numerical ordering. This is deliberately called a
*Dijkstra-style course heuristic*: the path label is a matrix-derived quantity,
not a sum of nonnegative scalar edge weights, so the ordinary scalar Dijkstra
optimality proof is not claimed. The retained `Sigma_p` is the course
approximation of relative-pose uncertainty between the two endpoints.

Candidate proximity uses an identity-measurement `BetweenFactorPose3` between
the optimized previous and current poses, with `Sigma_p` as its Gaussian
covariance. GTSAM factor error includes one half of the squared whitened
residual, so

```text
d_squared = mahalanobis_squared = 2 * factor_error
d = sqrt(d_squared).
```

Candidate detection returns scores and paths. It does not itself impose the
production temporal gate, threshold, matcher cap, or final acceptance policy.

## 5. Candidate production policy

The production policy is frozen in `ProductionPolicy`:

- A historical candidate must have an ordinal gap of at least 16. Equivalently,
  the previous 15 keyframe ordinals are excluded.
- The Mahalanobis comparison is strict: `d_squared < 36,000,000`, or `d < 6000`.
- Passing candidates are ordered by squared Mahalanobis score, previous
  keyframe ordinal, then previous FrameId.
- A persisted accepted-loop anchor suppresses candidates within inclusive
  `+/-10` previous ordinals and `+/-10` current ordinals. Rectangles are
  anchored and nontransitive.
- Remaining ranked candidates form temporary, likewise anchored proposal
  regions. These regions are scheduling devices, not persisted acceptance.
- Scheduling is region-first: one live head from each region is offered before
  a same-region fallback.
- At most three actual matcher invocations occur for a current keyframe.
  Scoring, thresholding, suppression, or merely considering a candidate does
  not consume the cap.

Detection is frozen once at the start of each current keyframe. This preserves
a complete, deterministic ranking and path record for that current even if one
of its candidates is accepted later in the schedule. An accepted same-current
loop immediately updates the Pose Graph and therefore supplies the latest pose
initialization to a later small Bundle, but it does not cause the frozen scores
to be recomputed. The next current keyframe detects afresh using the newest
optimized Values and the newest covariance graph.

## 6. Four-view consensus and RANSAC

For each actual matcher invocation, `evaluate_candidate_pair` resets the
Python, NumPy, and OpenCV random seeds to the pair seed, which is canonically
zero. Its evidence sets are successive and should not be conflated:

1. **Raw stereo matches** come from feature extraction and matching within the
   previous and current stereo pairs. The canonical front end uses SIFT/FLANN.
2. **Stereo inliers** are the raw matches that satisfy the established
   rectified stereo geometry at each endpoint.
3. **Reciprocal temporal matches** are mutually consistent descriptor matches
   between the previous-left and current-left images.
4. **Four-view consensus matches** are temporal matches that also map through
   both endpoint stereo-inlier sets, giving previous-left, previous-right,
   current-left, and current-right observations.
5. **Final supporters/inliers** are consensus entries whose reprojections are
   within two pixels in all four views under the retained pose model.

Previous-frame stereo inliers are triangulated in the local previous-camera
geometry. The custom RANSAC samples four consensus entries and performs one
PnP call per hypothesis, composes the right-camera poses from the fixed stereo
calibration, validates rigid geometry, and evaluates the four-view supporter
rule. Failed PnP calls and invalid models still consume attempts. The adaptive
requirement uses 99% success probability and the best observed supporter
ratio; the total budget is 3000 attempts.

After adaptive random sampling completes, the current supporter set is used
for refinement while it produces strict supporter-count increases. Two
terminal outcomes may proceed to the same production gates:

- `adaptive_completed` records that the adaptive requirement and refinement
  terminated within budget.
- `budget_exhausted_with_best_model` records that the fixed budget ended first
  but a complete best model and supporter mapping were retained.

Neither status alone means acceptance. The retained geometry must be complete,
and the inclusive gates require at least 30 four-view consensus matches, at
least 30 final supporters, and at least 30% supporters. Counts, attempt
accounting, status/reason, best geometry, supporter indices, and runtimes are
retained. Non-serialized keypoint and stereo mappings carry accepted supporter
observations into the small Bundle.

## 7. Small Bundle measurement

The small Bundle turns the image-level supporter set into a pose measurement
and covariance suitable for the cumulative Pose Graph. It intentionally
contains only:

- the previous loop-endpoint camera;
- the current loop-endpoint camera;
- one `q` landmark per eligible final supporter;
- one prior fixing the previous camera to identity; and
- two `GenericStereoFactor3D` factors per landmark, one at each endpoint.

There is no `BetweenFactorPose3` inside this Bundle. The latest cumulative Pose
Graph value `previous_pose.between(current_pose)` initializes only the current
camera. Each supporter contributes a canonical stereo measurement
`(u_left, u_right, mean(v_left, v_right))` at both endpoints.

Landmarks are initialized by backprojecting the previous stereo measurement.
Measurements with nonfinite values or nonpositive disparity are rejected, as
are nonfinite points, nonpositive depth, excessive initialization depth, or
points not in front of the initialized current camera. After optimization, any
point behind either camera causes a deterministic rebuild without that
landmark. All rejection reasons and rebuild history remain evidence.

Optimization proceeds in the previous-camera gauge:

1. LM optimizes the graph initialized with the previous camera at identity and
   the current camera at the latest Pose Graph relative pose.
2. Every optimized camera and landmark is exactly re-expressed relative to the
   optimized previous camera.
3. LM runs again from that normalized representation.
4. The result is normalized once more so the previous camera is exactly the
   local identity.

The measurement is then extracted as
`optimized_previous.between(optimized_current)`. Composition and point-mapping
checks verify that it is `T_Cprevious_from_Ccurrent`.

GTSAM Marginals supplies the ordered 12x12 joint covariance and information
for `[previous, current]`. If the covariance is partitioned into 6x6 blocks,
the conditional current/end covariance given previous/start is the symmetric
Schur complement

```text
Sigma_current|previous
    = Sigma_11 - Sigma_10 * solve(Sigma_00, Sigma_01).
```

The inverse of the matching current/current information block is retained as a
cross-check. The conditional matrix must be finite, symmetric within the
frozen tolerance, positive definite, and directly accepted by
`gtsam.noiseModel.Gaussian.Covariance`. A standalone factor cross-check confirms
the pose, direction, residual basis, and covariance contract. This is raw
scientific covariance in rotation-radian/translation-metre tangent order; no
visualization multiplier or other covariance modification is applied.

## 8. Cumulative Pose Graph update

The baseline loader rebuilds the consecutive E6 graph from the explicitly
selected E5 root and verifies its saved optimized Values and Marginals. The
explicit root is essential: a graph reconstructed from one E5 lane must not be
combined with E6 numerical arrays from another.

`CumulativePoseGraphState` holds the latest fully accepted graph, optimized
Values, pose matrices, all-node Marginals, and a state version. At version zero
the graph has 356 factors: 355 consecutive factors plus the frame-0 prior.
Every accepted Loop Closure increments the version and factor count by exactly
one.

For an accepted measurement, `_compute_loop_factor_update`:

1. validates the raw 6x6 conditional covariance;
2. creates one previous-to-current `BetweenFactorPose3` with that covariance;
3. appends it to a copy of the cumulative graph;
4. immediately runs Levenberg-Marquardt optimization;
5. re-expresses all poses relative to optimized frame 0, preserving the global
   identity gauge;
6. validates graph error and motion evidence;
7. reconstructs and validates Marginals for all 356 camera nodes; and
8. returns a complete proposed state.

`apply_loop_factor` is the publication boundary. It assigns the graph, Values,
pose matrices, Marginals, and new state version only after the proposed update
has succeeded in full. A factor, optimizer, gauge, or Marginals failure leaves
the last published state unchanged.

After successful integration, the same raw conditional covariance becomes a
directed accepted edge in the candidate-detection graph. It affects path
detection for future current keyframes; it does not retroactively rescore the
current keyframe's frozen candidates.

## 9. Sequential transaction and sealing

`_process_current_keyframe` makes the complete production sequence visible in
one place. Its phase comments identify:

1. the immutable current detection snapshot;
2. ordered-pair enumeration;
3. temporal exclusion;
4. covariance-path/Mahalanobis scoring;
5. strict thresholding and ranking;
6. durable accepted-region suppression;
7. temporary proposal regions;
8. fair region-first scheduling;
9. actual matcher invocation and its maximum-three accounting;
10. the consensus/RANSAC policy decision;
11. two-pose small-Bundle measurement construction;
12. candidate-local measurement failure;
13. one-factor cumulative Pose Graph update;
14. immediate optimization and all-node Marginals;
15. accepted raw covariance-edge publication for future detection;
16. accepted suppression-anchor publication;
17. complete artifact/evidence assembly; and
18. validation and atomic current-keyframe sealing.

The main state categories are:

- **Scientific state:** cumulative graph, optimized Values, pose matrices,
  Marginals, and directed covariance graph.
- **Candidate-local temporary state:** score row, matcher/RANSAC result, small
  Bundle, prospective edge, and the current staging directory.
- **Persisted accepted state:** a fully validated factor update, covariance
  edge, acceptance ID/state version, and suppression anchor.
- **Sealed artifact state:** the validated, atomically renamed current
  directory. This is the only durable resume state.

A normal matcher rejection or candidate-local measurement failure retains its
terminal evidence but publishes no factor, covariance edge, or suppression
anchor. In particular, the two canonical `measurement_failed` candidates did
not mutate scientific state. Persistent cluster suppression exists only after
successful scientific integration; a temporary proposal region never creates
it.

Accepted numerical evidence is first written under a temporary current
directory. The complete candidate dispositions and state counts are validated,
then the directory is atomically renamed to its final seal. A fatal failure
removes the attempt directory and stops the run. On restart, the runner
validates and replays the contiguous seal prefix, including accepted factors,
covariance edges, anchors, and the evaluated-pair set. It never treats an
unsealed directory or stale in-memory object as durable progress.

## 10. Canonical clean result

The clean canonical result under `diagnostics/ex7_clean_final` contains:

- 356 keyframes;
- 63,190 ordered historical/current pairs;
- 1,002 actual matcher invocations;
- 6 accepted Loop Closures;
- 2 candidate-local measurement failures;
- 362 final Pose Graph factors (356 baseline plus 6 loop factors); and
- 356 final camera Marginals.

The six accepted revisits, in acceptance order, are:

| Acceptance | Previous -> current FrameId | Consensus | Supporters | Supporter % |
| ---: | --- | ---: | ---: | ---: |
| 1 | 63 -> 1478 | 140 | 51 | 36.42857142857143% |
| 2 | 327 -> 2365 | 192 | 85 | 44.270833333333336% |
| 3 | 2266 -> 3206 | 445 | 347 | 77.97752808988764% |
| 4 | 2355 -> 3295 | 176 | 62 | 35.22727272727273% |
| 5 | 289 -> 3307 | 143 | 43 | 30.06993006993007% |
| 6 | 372 -> 3359 | 147 | 46 | 31.292517006802722% |

All six terminated as `adaptive_completed`, but their acceptance depended on
complete retained geometry and the 30/30/30 gates, not status alone. Each edge
records recognition of previously viewed content/location.

For context, sealed report metrics show mean/RMSE/final absolute location error
changing from 20.04525655083928 / 26.458709861447154 /
13.433460351647597 m without Loop Closure to 1.8167513809158942 /
2.0611940512492097 / 1.3768057126864905 m with Loop Closure. Mean/RMSE/final
absolute rotation error changes from 6.126462346031795 /
7.15089634249872 / 11.387075625240138 degrees to 0.9044403077423985 /
1.1237217726219766 / 0.5627874255110284 degrees. These are results for the
frozen course pipeline on KITTI 00, not a cross-sequence generalization claim.

## 11. Uncertainty and covariance visualizations

Four related quantities must remain distinct.

### A. Scientific covariance

The scientific objects are the raw saved 6x6 Pose3 Marginals and raw 6x6
conditional Loop Closure covariances. They remain in the validated local
rotation-then-translation tangent basis and are not changed for reporting.

### B. Scalar location uncertainty

For a marginal covariance `Sigma`, the displayed scalar location size is

```text
sqrt(trace(Sigma[3:6, 3:6]))  [m].
```

It summarizes the three translation variances. It is not a confidence ellipse
and does not change `Sigma`.

### C. Scalar angular uncertainty

The angular size uses

```text
sqrt(trace(Sigma[0:3, 0:3]))  [rad],
```

with conversion to degrees only where the report unit is degrees. It likewise
summarizes a raw marginal block rather than modifying it.

### D. Top-down 95% location covariance ellipse

For pose rotation `R`, reporting performs:

```text
local translation block Sigma_t
    -> global translation covariance R * Sigma_t * R^T
    -> X-Z 2x2 block
    -> eigendecomposition
    -> semi-axis radii sqrt(chi2_2d_95 * eigenvalues).
```

The two-dimensional 95% chi-square value is approximately 5.99146. “Raw” means
the saved covariance was not multiplied, repaired, regularized, or otherwise
changed. “95%” describes the confidence contour derived for presentation.
Displaying every twelfth keyframe plus loop endpoints is subsampling of drawn
ellipses only. A future x5/x10 ellipse-radius variant would also be explicitly
presentation-only; it would not be a scientific covariance change.

## 12. Clean-run validation/equivalence

The relevant roots are:

- `diagnostics/ex7_clean_final`: accepted clean canonical scientific result;
- `diagnostics/ex7_i04`: protected historical E7-I04 reference; and
- `diagnostics/ex7_clean_equivalence`: independent comparison of their stored
  evidence.

The comparison found the same 356 keyframes, 63,190 ordered pairs, 1,002
matcher invocations, candidate dispositions, six accepted loops, state/factor
progression, and final 356 Marginals. All 356 candidate-path NPZ files and all
six accepted checkpoint NPZ files were byte-identical, as were the common
scientific report CSVs. Accepted-checkpoint and final pose/marginal comparisons
had zero observed numerical difference. The clean result is therefore the
canonical Exercise 7 scientific evidence for later documentation and reporting.

## 13. Report-only versus scientific work

The scientific boundary includes:

- feature/stereo/temporal matching;
- bounded PnP/RANSAC and supporter classification;
- two-pose small-Bundle construction and optimization;
- Loop Closure factor insertion and cumulative Pose Graph optimization; and
- Marginals construction.

The report-only boundary includes:

- reading and aligning sealed E5/E6/E7 evidence;
- plotting already computed trajectories, errors, match counts, and
  uncertainty;
- converting raw covariance to a displayed 95% ellipse;
- computing scalar summaries from saved arrays; and
- building a curated submission package with provenance and checksums.

`exercise7_report_freeze.py` stays entirely on the second side of this
boundary. It cannot turn a reporting request into new Exercise 7 science.

## 14. Where to start reading

A concise reading order for a grader or student is:

1. Read `ex7.py` to see the safe command surface and selected-root flow.
2. Read the module overview, `ProductionPolicy`, and
   `_process_current_keyframe` phase comments in
   `exercise7_sequential_loop_closure.py` for the complete pipeline.
3. Follow one candidate through `exercise7_candidate_detection.py` and then
   `exercise7_consensus_matching.py`.
4. Study `solve_small_bundle` in `exercise7_small_bundle.py`, especially its
   direction checks and conditional covariance extraction.
5. Read `CumulativePoseGraphState.apply_loop_factor` and
   `_compute_loop_factor_update` in `exercise7_pose_graph_update.py` for the
   scientific-computation/publication boundary.
6. Finish with `exercise7_report_freeze.py` to see how sealed scientific
   evidence becomes presentation and submission material without a rerun.

Exact source line numbers are intentionally omitted because traceability line
locations should be refreshed only after the final documentation/reporting
source layout is frozen.
