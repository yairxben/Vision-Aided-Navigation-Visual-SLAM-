# Project Submission final requirements traceability matrix

Canonical source: `docs/submission/Project Submission - Ex8.pdf` (all seven pages locally extracted and audited).

Classifications are final-report honest: **PASS** means the requested evidence is materialized and validated; **OPTIONAL** is conditional/suggested work not required for this artifact freeze; **MISSING** requires final authored report/layout work; **NOT APPLICABLE** means the condition does not apply.

| ID | PDF page/section | Category | Stage | Classification | Requirement | Evidence | Remaining work |
|---|---|---|---|---|---|---|---|
| P1-01 | 1 | mandatory | overall report | MISSING | Explain the system stages in the authors' own words and summarize the algorithms as understood. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P1-02 | 1 §1 | mandatory | overall report | MISSING | Provide an introduction, background, and explanation of why visual navigation is important. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P1-03 | 1 §2 | mandatory | overall report | MISSING | Explain code organization and the functions implementing the main stages; code should be readable and documented. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P1-C01 | 1 §2 | code-location | Tracking | PASS | Supply function, file, and line location for triangulation. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P1-C02 | 1 §2 | code-location | PnP | PASS | Supply function, file, and line location for RANSAC. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P1-C03 | 1 §2 | code-location | PnP | PASS | Supply function, file, and line location for PnP trajectory calculation. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P1-C04 | 1 §2 | code-location | Tracking | PASS | Supply function/class, file, and line location for DataBase definition. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P1-C05 | 1 §2 | code-location | Tracking | PASS | Supply function, file, and line location for adding a frame to the database. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P1-C06 | 1 §2 | code-location | Bundle | PASS | Supply function, file, and line location for Single Bundle creation. | `bundle_clean_final/`; `final_report/figures/bundle_*` | None for the artifact package. |
| P1-C07 | 1 §2 | code-location | Bundle | PASS | Supply location for relative transformation, including covariance, extraction from a bundle result. | `bundle_clean_final/`; `final_report/figures/bundle_*` | None for the artifact package. |
| P1-C08 | 1 §2 | code-location | Pose Graph | PASS | Supply function, file, and line location for PoseGraph building. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P1-C09 | 1 §2 | code-location | Loop Closure | PASS | Supply locations for loop-closure detection and factor creation. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P1-04 | 1 §3 | mandatory | overall report | MISSING | Provide detailed quantitative and qualitative performance analysis for the system stages. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P1-05 | 1 §3 | conditional | overall report | MISSING | Where relevant, describe weak-point improvements, the implemented change, and its impact. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P1-06 | 1 §3 | suggested | overall report | PASS | Use logarithmic scale where appropriate, such as track-length or covariance graphs. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P1-07 | 1 §3 | suggested | overall report | PASS | Overlay data where appropriate for easy comparison. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P1-08 | 1 §3 | normative | overall report | PASS | Use the same scale for comparable graphs. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P1-09 | 1 §4 | mandatory | overall report | MISSING | Summarize, criticize weak points and unrealistic assumptions, propose improvements and future research. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P1-10 | 1 §4 | conditional | overall report | OPTIONAL | Describe extra research/implementation, motivation, and results if applicable. | `final_report/`; `code_locations.json`; stage packages | Optional editorial selection. |
| P2-01 | 2 | report-format | overall report | MISSING | Limit the final report to under 20 pages. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P2-02 | 2 | report-format | overall report | MISSING | Number all pages and all figures. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P2-03 | 2 | report-format | overall report | MISSING | Specify exact code location that created each figure next to the figure. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P2-04 | 2 | normative | overall report | PASS | Label every figure axis with units. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P2-05 | 2 | normative | overall report | PASS | Use relevant scale, meaningful zoom/crop, colors, and effective overlays for clear figures. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P2-06 | 2 Performance Analysis | mandatory | overall report | PASS | Present evidence for each stage and the system as a whole. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P2-07 | 2 Performance Analysis | mandatory | overall report | MISSING | Analyze every graph, highlighting what it shows and positive or negative behavior. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P2-08 | 2 Performance Analysis | normative | overall report | MISSING | Combine related data, crop meaningfully, and number each figure. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P2-T01 | 2 Tracking | mandatory | Tracking | PASS | Report total number of tracks. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P2-T02 | 2 Tracking | mandatory | Tracking | PASS | Report number of frames. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P2-T03 | 2 Tracking | mandatory | Tracking | PASS | Report mean track length. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P2-T04 | 2 Tracking | mandatory | Tracking | PASS | Report mean number of frame Links. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P2-T05 | 2 Tracking | mandatory | Tracking | PASS | Graph number of matches per frame. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P2-T06 | 2 Tracking | mandatory | Tracking | PASS | Graph percentage of inliers per frame. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P2-S01 | 2 Suggested figures | suggested | overall report | PASS | Include a substantial subset of suggested figures and any useful additions. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P2-S02 | 2 Suggested figures | suggested | Tracking | PASS | Plot connectivity: tracks in each frame with Links in the next frame. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P3-01 | 3 | suggested | Tracking | PASS | Plot a track-length histogram. | `tracking/`; audit PASS, unchanged | None for the artifact package. |
| P3-02 | 3 | suggested | overall report | PASS | Overlay bird's-eye PnP, Bundle, Pose Graph with loop closure, and ground truth. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P3-03 | 3 | suggested | Bundle | PASS | Plot bundle mean factor error per starting keyframe before and after optimization. | `bundle_clean_final/`; `final_report/figures/bundle_*` | None for the artifact package. |
| P3-04 | 3 | suggested | Bundle | PASS | Plot bundle median projection error before and after optimization. | `bundle_clean_final/`; `final_report/figures/bundle_*` | None for the artifact package. |
| P3-05 | 3 | suggested | PnP | PASS | Plot meaningful projection-error statistic versus distance from PnP triangulation frame. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P3-06 | 3 | suggested | Bundle | PASS | Plot Bundle projection error versus distance from the first frame. | `bundle_clean_final/`; `final_report/figures/bundle_*` | None for the artifact package. |
| P3-07 | 3 note | conditional | overall report | PASS | If projection collection uses a representative subset, explain its selection. | `final_report/`; `code_locations.json`; stage packages | None for the artifact package. |
| P4-01 | 4 | suggested | PnP | PASS | Plot absolute PnP X/Y/Z signed errors and total location-error norm in meters. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P4-02 | 4 | suggested | PnP | PASS | Plot absolute PnP angle error in degrees. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P4-03 | 4 | suggested | Pose Graph | PASS | Plot absolute Pose Graph error without loop closure: X/Y/Z, location norm, and angle. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P4-04 | 4 | suggested | Loop Closure | PASS | Plot absolute Pose Graph error with loop closure: X/Y/Z, location norm, and angle. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P4-05 | 4 Relative Error | suggested | Bundle | PASS | Compare consecutive-keyframe Bundle and accumulated PnP relative pose errors. | `bundle_clean_final/`; `final_report/figures/bundle_*` | None for the artifact package. |
| P4-06 | 4 Relative Error | mandatory-discussion | overall report | MISSING | Discuss whether relative errors suggest inaccuracies in ground-truth poses. | `final_report/`; `code_locations.json`; stage packages | Author/place in the final under-20-page numbered report. |
| P5-01 | 5 | suggested | PnP | PASS | Evaluate relative PnP subsections of lengths 100, 400, and 800. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P5-02 | 5 | suggested | PnP | PASS | For each PnP length, report translation error per meter/error percent, angle error per meter, and averages. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P5-03 | 5 | suggested | Bundle | PASS | Evaluate relative Bundle subsections at 100/400/800, choosing closest keyframe if needed, with normalized errors and averages. | `bundle_clean_final/`; `final_report/figures/bundle_*` | None for the artifact package. |
| P5-04 | 5 | suggested | Loop Closure | PASS | Plot matches per successful loop-closure frame. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P5-05 | 5 | suggested | Loop Closure | PASS | Plot inlier percentage per successful loop-closure frame. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P5-06 | 5 | suggested | Pose Graph | PASS | Plot location and angle uncertainty versus keyframe without loop closure. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P6-01 | 6 | suggested | Loop Closure | PASS | Plot location and angle uncertainty versus keyframe with loop closure. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P6-02 | 6 | mandatory-discussion | Pose Graph | PASS | Explain how uncertainty size was measured and how location/angle components were isolated. | `loop_closure_clean_final/`; `final_report/figures/pose_graph_*` | None for the artifact package. |
| P6-A01 | 6 Appendix | normative | PnP | PASS | Measure rotation magnitude with Rodrigues-vector norm converted to degrees. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P6-A02 | 6 Appendix | normative | PnP | PASS | Compare estimated and GT relative poses; use consecutive frames or keyframes as appropriate. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P6-A03 | 6 Appendix | normative | PnP | PASS | For long subsections, report error per meter traveled as in KITTI evaluation. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P6-A04 | 6 Appendix | normative | PnP | PASS | For a length-100 example such as 51→150, compare estimated and true movement and divide by traveled path length. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P7-A01 | 7 Appendix | normative | PnP | PASS | Location numerator is the norm of the translation part of the relative-pose error. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P7-A02 | 7 Appendix | normative | PnP | PASS | Angle numerator is the rotation-part magnitude of the relative-pose error. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P7-A03 | 7 Appendix | normative | PnP | PASS | Denominator is summed inter-frame GT path distance, not endpoint displacement, and is shared by location and angle. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P7-A04 | 7 Appendix | normative | PnP | PASS | Use all sliding windows: for length 100, 0→99, 1→100, …, (n-99)→n. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |
| P7-A05 | 7 Appendix | normative | PnP | PASS | Respect appendix pose, relative-pose, displacement, error-pose, translation-norm, rotation-norm, and total-distance definitions. | `pnp/`; `final_report/figures/pnp_*` | None for the artifact package. |

## Honest remaining boundary

The evidence package is complete for quantitative stage reporting. The remaining MISSING rows are authored narrative, final document placement/numbering, the rendered page-count check, figure-adjacent code citations, and the requested ground-truth-accuracy discussion. This phase does not fabricate those editorial deliverables.
