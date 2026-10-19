"""Thin Exercise 6 runner for the consecutive-keyframe Pose Graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from PoseGraph import (
    CANONICAL_INPUT_ROOT,
    CanonicalPoseGraphInputs,
    PoseGraph,
    load_canonical_pose_graph_inputs,
)
from exercise6_reporting import (
    create_visualization_only_covariance_figures as _create_visualizations,
    write_pose_graph_outputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "diagnostics" / "ex6_clean_final"
)


def create_visualization_only_covariance_figures(
    *,
    input_root: Path = CANONICAL_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, object]:
    """Create only the optional visualization-scaled covariance figures."""

    return _create_visualizations(
        input_root=input_root,
        output_root=output_root,
    )


def run_canonical_pose_graph(
    *,
    input_root: Path = CANONICAL_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, object]:
    """Run the unchanged Exercise 6 algorithm from one sealed E5 root."""

    # Load the ordered consecutive Bundle constraints and their raw
    # conditional end-given-start covariances.
    inputs = load_canonical_pose_graph_inputs(input_root)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    # Construct one anchor prior, all consecutive BetweenFactorPose3 factors,
    # and the composed initial trajectory.
    pose_graph = PoseGraph.from_canonical_inputs(inputs)
    initial_error = float(
        pose_graph.graph().error(pose_graph.get_initial_estimate())
    )

    # Optimize once with the default GTSAM Levenberg-Marquardt optimizer.
    optimized_values = pose_graph.optimize()
    final_error = float(pose_graph.graph().error(optimized_values))
    max_translation_m, max_rotation_rad = (
        pose_graph.optimization_movement()
    )
    _validate_optimization(
        inputs=inputs,
        initial_error=initial_error,
        final_error=final_error,
        max_translation_m=max_translation_m,
        max_rotation_rad=max_rotation_rad,
    )

    # Compute standard GTSAM Marginals from the same graph and optimized
    # Values.
    marginals = pose_graph.compute_marginals()
    marginal_covariances = pose_graph.marginal_covariance_array()
    if (
        marginal_covariances.shape != (len(inputs.keyframe_ids), 6, 6)
        or not np.all(np.isfinite(marginal_covariances))
    ):
        raise RuntimeError("Pose Graph Marginals are missing or nonfinite")

    # Serialize the unchanged scientific arrays and generate report outputs.
    return write_pose_graph_outputs(
        inputs=inputs,
        pose_graph=pose_graph,
        optimized_values=optimized_values,
        marginals=marginals,
        marginal_covariances=marginal_covariances,
        initial_error=initial_error,
        final_error=final_error,
        max_translation_m=max_translation_m,
        max_rotation_rad=max_rotation_rad,
        output_root=output_root,
    )


def _validate_optimization(
    *,
    inputs: CanonicalPoseGraphInputs,
    initial_error: float,
    final_error: float,
    max_translation_m: float,
    max_rotation_rad: float,
) -> None:
    """Apply the existing E6 numerical validation tolerances unchanged."""

    tolerances = inputs.validation_tolerances
    allowed_error_increase = max(
        tolerances["graph_error_nonincrease_absolute"],
        tolerances["graph_error_nonincrease_relative"] * abs(initial_error),
    )
    if final_error > initial_error + allowed_error_increase:
        raise RuntimeError("Pose Graph optimization increased graph error")

    movement_tolerance = tolerances["numerical_reconstruction_atol"]
    if (
        max_translation_m > movement_tolerance
        or max_rotation_rad > movement_tolerance
    ):
        raise RuntimeError(
            "Pose Graph optimization caused meaningful trajectory movement"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Exercise 6 from sealed consecutive Exercise 5 endpoints"
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=CANONICAL_INPUT_ROOT,
        help=(
            "sealed E5 root whose embedded canonical-root contract matches "
            "the selected directory (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="new output directory (must not already exist)",
    )
    parser.add_argument(
        "--create-visualization-x0.01-only",
        dest="create_visualization_x001_only",
        action="store_true",
        help=(
            "add only the two covariance ×0.01 visualization copies "
            "to an existing E6 output root"
        ),
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.create_visualization_x001_only:
        result = create_visualization_only_covariance_figures(
            input_root=arguments.input_root,
            output_root=arguments.output_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    summary = run_canonical_pose_graph(
        input_root=arguments.input_root,
        output_root=arguments.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
