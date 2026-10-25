"""Analysis-only KITTI pose loading.

Estimation modules must not import this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import ContractError
from ..geometry import validate_rigid_pose


WORLD_TO_CAMERA = "world-to-camera"
CAMERA_TO_WORLD = "camera-to-world"
GROUND_TRUTH_CONVENTIONS = (WORLD_TO_CAMERA, CAMERA_TO_WORLD)


def normalize_ground_truth_convention(value: str) -> str:
    convention = str(value).strip().lower()
    if convention not in GROUND_TRUTH_CONVENTIONS:
        raise ContractError(
            "GT convention must be one of: {}".format(
                ", ".join(GROUND_TRUTH_CONVENTIONS)
            )
        )
    return convention


def load_ground_truth_world_to_camera(
    path: Path,
    expected_count: int,
    input_convention: str = WORLD_TO_CAMERA,
) -> np.ndarray:
    """Load GT and normalize it to the project ``T_C_from_W`` convention.

    Course/lecturer pose files are already world-to-camera and therefore use
    the default. Original KITTI odometry pose rows are camera-to-world and must
    be selected explicitly; no sequence identity is used to infer direction.
    """

    convention = normalize_ground_truth_convention(input_convention)
    try:
        lines = [
            line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except OSError as error:
        raise ContractError("cannot read ground truth: {}".format(error))
    if len(lines) != expected_count:
        raise ContractError(
            "ground-truth row count {} differs from frame count {}".format(
                len(lines), expected_count
            )
        )
    poses = []
    for index, line in enumerate(lines):
        try:
            pose = np.asarray(line.split(), dtype=float).reshape(3, 4)
        except ValueError as error:
            raise ContractError("invalid ground-truth row {}: {}".format(index, error))
        validated = validate_rigid_pose(pose, atol=1e-6)
        if convention == CAMERA_TO_WORLD:
            validated = validate_rigid_pose(np.linalg.inv(validated), atol=1e-6)
        poses.append(validated)
    return np.asarray(poses, dtype=float)


__all__ = [
    "CAMERA_TO_WORLD",
    "GROUND_TRUTH_CONVENTIONS",
    "WORLD_TO_CAMERA",
    "load_ground_truth_world_to_camera",
    "normalize_ground_truth_convention",
]
