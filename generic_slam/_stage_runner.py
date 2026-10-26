"""Private child-process dispatcher used by the sole public CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .artifacts.io import read_json
from .artifacts.manifest import load_stage_manifest, validate_stage_manifest
from .config import SCIENTIFIC_IMPLEMENTATION_REVISION, load_resolved_config
from .contracts import ContractError, GenericSlamError
from .dataset.kitti import discover_kitti_estimation_context, discover_kitti_sequence


STAGES = (
    "tracking", "keyframes", "bundle", "pose_graph", "loop_closure",
    "analysis", "report",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gt-convention", choices=("world-to-camera", "camera-to-world")
    )
    return parser


def _verified_manifest(
    root: Path,
    stage: str,
    sequence_id: str,
    dataset_id: str,
    config_fingerprint: str,
    upstream_stage_ids: Optional[Mapping[str, str]] = None,
) -> object:
    manifest = load_stage_manifest(Path(root) / "manifest.json")
    validate_stage_manifest(
        manifest,
        root,
        expected_stage=stage,
        sequence_id=sequence_id,
        dataset_id=dataset_id,
        config_fingerprint=config_fingerprint,
        upstream_stage_ids=(
            manifest.upstream_stage_ids if upstream_stage_ids is None else upstream_stage_ids
        ),
    )
    return manifest


def _ids(**manifests: object) -> Mapping[str, str]:
    return {name: manifest.stage_id for name, manifest in manifests.items()}


def run_stage(argv: Sequence[str]) -> None:
    arguments = _parser().parse_args(argv)
    output = arguments.output_root.expanduser().resolve()
    run = read_json(output / "run.json")
    if not isinstance(run, dict):
        raise ContractError("run.json must be an object")
    if run.get("scientific_implementation_revision") != SCIENTIFIC_IMPLEMENTATION_REVISION:
        raise ContractError("child stage scientific implementation revision differs from run")
    if arguments.stage in ("analysis", "report"):
        from .dataset.ground_truth import normalize_ground_truth_convention

        run_convention = normalize_ground_truth_convention(
            str(run.get("gt_input_convention", "world-to-camera"))
        )
        if arguments.gt_convention is None or arguments.gt_convention != run_convention:
            raise ContractError("analysis GT convention differs from managed run")
        dataset = discover_kitti_sequence(
            arguments.dataset_root,
            arguments.sequence,
            ground_truth_convention=run_convention,
        )
        context = dataset.estimation_context()
    else:
        if arguments.gt_convention is not None:
            raise ContractError("GT convention is not accepted by estimation stages")
        context = discover_kitti_estimation_context(
            arguments.dataset_root, arguments.sequence
        )
        dataset = context
    if (
        run.get("sequence_id") != dataset.sequence_id
        or run.get("dataset_id") != dataset.dataset_id
        or run.get("dataset_root") != str(dataset.dataset_root)
    ):
        raise ContractError("child stage dataset identity differs from run")
    config = load_resolved_config(output / "resolved_config.json", str(run.get("run_mode")))
    if config.fingerprint != run.get("config_fingerprint"):
        raise ContractError("child stage configuration identity differs from run")
    dataset_manifest = _verified_manifest(
        output / "dataset", "dataset", dataset.sequence_id, dataset.dataset_id,
        config.stage_fingerprint("dataset"), {},
    )

    # Reporting deliberately depends only on durable manifests and analysis
    # artifacts. It remains runnable even when estimator modules are disabled.
    if arguments.stage == "report":
        tracking_manifest = _verified_manifest(
            output / "tracking", "tracking", dataset.sequence_id, dataset.dataset_id,
            config.stage_fingerprint("tracking"), _ids(dataset=dataset_manifest),
        )
        keyframe_manifest = _verified_manifest(
            output / "keyframes", "keyframes", dataset.sequence_id, dataset.dataset_id,
            config.stage_fingerprint("keyframes"), _ids(tracking=tracking_manifest),
        )
        bundle_manifest = _verified_manifest(
            output / "bundle", "bundle", dataset.sequence_id, dataset.dataset_id,
            config.stage_fingerprint("bundle"),
            _ids(tracking=tracking_manifest, keyframes=keyframe_manifest),
        )
        pose_graph_manifest = _verified_manifest(
            output / "pose_graph", "pose_graph", dataset.sequence_id, dataset.dataset_id,
            config.stage_fingerprint("pose_graph"), _ids(bundle=bundle_manifest),
        )
        loop_manifest = _verified_manifest(
            output / "loop_closure", "loop_closure", dataset.sequence_id,
            dataset.dataset_id, config.stage_fingerprint("loop_closure"),
            _ids(pose_graph=pose_graph_manifest),
        )
        analysis_upstream = _ids(
            dataset=dataset_manifest, tracking=tracking_manifest,
            keyframes=keyframe_manifest, bundle=bundle_manifest,
            pose_graph=pose_graph_manifest, loop_closure=loop_manifest,
        )
        analysis_manifest = _verified_manifest(
            output / "analysis", "analysis", dataset.sequence_id, dataset.dataset_id,
            config.stage_fingerprint("analysis"), analysis_upstream,
        )
        from .analysis.reporting import run_report_stage

        run_report_stage(
            dataset, config, output / "analysis", output / "report",
            upstream_stage_ids=_ids(analysis=analysis_manifest),
        )
        return

    if arguments.stage == "tracking":
        from .tracking.stage import run_tracking_stage

        run_tracking_stage(
            context, config, output / "tracking",
            upstream_stage_ids=_ids(dataset=dataset_manifest),
        )
        return

    tracking_upstream = _ids(dataset=dataset_manifest)
    tracking_manifest = _verified_manifest(
        output / "tracking", "tracking", dataset.sequence_id, dataset.dataset_id,
        config.stage_fingerprint("tracking"), tracking_upstream,
    )
    if arguments.stage == "keyframes":
        from .keyframes.stage import run_keyframe_stage
        from .tracking.stage import (
            load_tracking_trajectory_output,
            validate_tracking_trajectory_artifact,
        )

        validate_tracking_trajectory_artifact(output / "tracking", dataset.frame_count)
        tracking = load_tracking_trajectory_output(output / "tracking")
        run_keyframe_stage(
            context, config, tracking, output / "keyframes",
            upstream_stage_ids=_ids(tracking=tracking_manifest),
        )
        return

    keyframe_upstream = _ids(tracking=tracking_manifest)
    keyframe_manifest = _verified_manifest(
        output / "keyframes", "keyframes", dataset.sequence_id, dataset.dataset_id,
        config.stage_fingerprint("keyframes"), keyframe_upstream,
    )
    if arguments.stage == "loop_closure":
        bundle_manifest = _verified_manifest(
            output / "bundle", "bundle", dataset.sequence_id, dataset.dataset_id,
            config.stage_fingerprint("bundle"),
            _ids(tracking=tracking_manifest, keyframes=keyframe_manifest),
        )
        pose_graph_manifest = _verified_manifest(
            output / "pose_graph", "pose_graph", dataset.sequence_id, dataset.dataset_id,
            config.stage_fingerprint("pose_graph"), _ids(bundle=bundle_manifest),
        )
        from .pose_graph.stage import load_pose_graph_output, validate_pose_graph_artifacts
        from .loop_closure.stage import run_loop_closure_stage

        expected_keyframes = int(pose_graph_manifest.counts["keyframe_count"])
        validate_pose_graph_artifacts(output / "pose_graph", expected_keyframes)
        pose_graph = load_pose_graph_output(output / "pose_graph")
        run_loop_closure_stage(
            context, config, pose_graph, output / "loop_closure",
            upstream_stage_ids=_ids(pose_graph=pose_graph_manifest),
        )
        return
    from .keyframes.stage import load_keyframe_output, validate_keyframe_artifacts

    validate_keyframe_artifacts(output / "keyframes", dataset.final_frame_id)
    keyframes = load_keyframe_output(output / "keyframes")
    if arguments.stage == "bundle":
        from .bundle.stage import run_bundle_stage
        from .tracking.stage import load_tracking_output, validate_tracking_artifacts

        database = validate_tracking_artifacts(output / "tracking", dataset.frame_count)
        tracking = load_tracking_output(
            output / "tracking", tracking_manifest, database=database
        )
        run_bundle_stage(
            context, config, tracking, keyframes, output / "bundle",
            upstream_stage_ids=_ids(tracking=tracking_manifest, keyframes=keyframe_manifest),
        )
        return

    bundle_upstream = _ids(tracking=tracking_manifest, keyframes=keyframe_manifest)
    bundle_manifest = _verified_manifest(
        output / "bundle", "bundle", dataset.sequence_id, dataset.dataset_id,
        config.stage_fingerprint("bundle"), bundle_upstream,
    )
    from .bundle.stage import load_bundle_output, validate_bundle_artifacts

    validate_bundle_artifacts(
        output / "bundle", keyframes.selection.frame_ids,
        dataset=context, config=config,
        upstream_stage_ids=bundle_upstream,
    )
    bundle = load_bundle_output(output / "bundle", bundle_manifest)
    if arguments.stage == "pose_graph":
        from .pose_graph.stage import run_pose_graph_stage, validate_pose_graph_artifacts

        existing_manifest = output / "pose_graph" / "manifest.json"
        if existing_manifest.is_file():
            _verified_manifest(
                output / "pose_graph", "pose_graph", dataset.sequence_id,
                dataset.dataset_id, config.stage_fingerprint("pose_graph"),
                _ids(bundle=bundle_manifest),
            )
            validate_pose_graph_artifacts(output / "pose_graph", len(bundle.keyframe_ids))
            return
        run_pose_graph_stage(
            context, config, bundle, output / "pose_graph",
            upstream_stage_ids=_ids(bundle=bundle_manifest),
        )
        return

    pose_graph_upstream = _ids(bundle=bundle_manifest)
    pose_graph_manifest = _verified_manifest(
        output / "pose_graph", "pose_graph", dataset.sequence_id, dataset.dataset_id,
        config.stage_fingerprint("pose_graph"), pose_graph_upstream,
    )
    from .pose_graph.stage import (
        load_pose_graph_artifacts,
        validate_pose_graph_artifacts,
    )

    validate_pose_graph_artifacts(output / "pose_graph", len(bundle.keyframe_ids))
    pose_graph = load_pose_graph_artifacts(output / "pose_graph")
    loop_upstream = _ids(pose_graph=pose_graph_manifest)
    loop_manifest = _verified_manifest(
        output / "loop_closure", "loop_closure", dataset.sequence_id, dataset.dataset_id,
        config.stage_fingerprint("loop_closure"), loop_upstream,
    )
    from .loop_closure.stage import load_loop_closure_output, validate_loop_closure_artifacts

    validate_loop_closure_artifacts(output / "loop_closure", pose_graph.keyframe_ids)
    loop_closure = load_loop_closure_output(output / "loop_closure")
    if arguments.stage == "analysis":
        from .analysis.reporting import run_analysis_stage
        from .tracking.stage import (
            load_tracking_trajectory_output,
            validate_tracking_trajectory_artifact,
        )

        validate_tracking_trajectory_artifact(output / "tracking", dataset.frame_count)
        tracking = load_tracking_trajectory_output(output / "tracking")
        run_analysis_stage(
            dataset, config, tracking, keyframes, bundle, pose_graph, loop_closure,
            output / "analysis",
            upstream_stage_ids=_ids(
                dataset=dataset_manifest, tracking=tracking_manifest,
                keyframes=keyframe_manifest, bundle=bundle_manifest,
                pose_graph=pose_graph_manifest, loop_closure=loop_manifest,
            ),
        )
        return

    raise ContractError("unreachable stage dispatch")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        run_stage(argv)
    except GenericSlamError as error:
        print("generic_slam internal stage: {}".format(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
