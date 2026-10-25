"""Runtime discovery of a KITTI odometry stereo sequence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..contracts import ContractError
from .calibration import StereoCalibration, load_stereo_calibration


CANONICAL_IMAGE_NAME = re.compile(r"^([0-9]{6})\.png$")


def normalize_sequence_id(value: str) -> str:
    text = str(value).strip()
    if not text or not text.isdigit():
        raise ContractError("KITTI sequence ID must contain only digits")
    return text.zfill(2)


def _discover_image_directory(path: Path) -> Tuple[Dict[int, Path], Tuple[str, ...]]:
    if not path.is_dir():
        raise ContractError("KITTI image directory is missing: {}".format(path))
    images: Dict[int, Path] = {}
    ignored = []
    for item in sorted(path.iterdir(), key=lambda candidate: candidate.name):
        match = CANONICAL_IMAGE_NAME.fullmatch(item.name)
        if item.is_file() and match is not None:
            frame_id = int(match.group(1))
            if frame_id in images:
                raise ContractError("duplicate canonical FrameId {}".format(frame_id))
            images[frame_id] = item
        else:
            ignored.append(item.name)
    return images, tuple(ignored)


@dataclass(frozen=True)
class KittiStereoSequence:
    dataset_root: Path
    sequence_id: str
    sequence_root: Path
    frame_ids: Tuple[int, ...]
    left_images: Tuple[Path, ...]
    right_images: Tuple[Path, ...]
    ignored_left_entries: Tuple[str, ...]
    ignored_right_entries: Tuple[str, ...]
    calibration_path: Path
    calibration: StereoCalibration
    ground_truth_path: Optional[Path]
    ground_truth_world_to_camera: Optional[np.ndarray]
    ground_truth_input_convention: str
    dataset_id: str

    def estimation_context(self) -> "KittiEstimationContext":
        """Return the GT-free view accepted by every estimation stage."""

        return KittiEstimationContext(
            dataset_root=self.dataset_root,
            sequence_id=self.sequence_id,
            sequence_root=self.sequence_root,
            frame_ids=self.frame_ids,
            left_images=self.left_images,
            right_images=self.right_images,
            calibration_path=self.calibration_path,
            calibration=self.calibration,
            dataset_id=self.dataset_id,
        )

    @property
    def frame_count(self) -> int:
        return len(self.frame_ids)

    @property
    def final_frame_id(self) -> int:
        return self.frame_ids[-1]

    def left_image_path(self, frame_id: int) -> Path:
        self._validate_frame_id(frame_id)
        return self.left_images[frame_id]

    def right_image_path(self, frame_id: int) -> Path:
        self._validate_frame_id(frame_id)
        return self.right_images[frame_id]

    def read_stereo(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        left = cv2.imread(str(self.left_image_path(frame_id)), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(self.right_image_path(frame_id)), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise ContractError("cannot decode stereo images for FrameId {}".format(frame_id))
        return left, right

    def _validate_frame_id(self, frame_id: int) -> None:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ContractError("FrameId must be an integer")
        if not 0 <= frame_id < self.frame_count:
            raise ContractError("FrameId {} is outside this sequence".format(frame_id))

    def identity_record(self) -> Dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "dataset_root": str(self.dataset_root),
            "frame_ids": list(self.frame_ids),
            "calibration": self.calibration.record(),
        }

    def manifest_record(self) -> Dict[str, object]:
        return {
            "schema": "generic-slam-dataset-v1",
            "status": "complete",
            "dataset_id": self.dataset_id,
            "sequence_id": self.sequence_id,
            "dataset_root": str(self.dataset_root),
            "sequence_root": str(self.sequence_root),
            "frame_count": self.frame_count,
            "first_frame_id": self.frame_ids[0],
            "final_frame_id": self.final_frame_id,
            "ignored_left_entries": list(self.ignored_left_entries),
            "ignored_right_entries": list(self.ignored_right_entries),
            "ground_truth_available": self.ground_truth_path is not None,
            "ground_truth_path": (
                str(self.ground_truth_path) if self.ground_truth_path is not None else None
            ),
            "ground_truth_input_convention": self.ground_truth_input_convention,
            "ground_truth_internal_convention": "world-to-camera",
            "calibration_path": str(self.calibration_path),
        }


@dataclass(frozen=True)
class KittiEstimationContext:
    """Dataset geometry and images with no Ground Truth field or loader."""

    dataset_root: Path
    sequence_id: str
    sequence_root: Path
    frame_ids: Tuple[int, ...]
    left_images: Tuple[Path, ...]
    right_images: Tuple[Path, ...]
    calibration_path: Path
    calibration: StereoCalibration
    dataset_id: str

    @property
    def frame_count(self) -> int:
        return len(self.frame_ids)

    @property
    def final_frame_id(self) -> int:
        return self.frame_ids[-1]

    def _validate_frame_id(self, frame_id: int) -> None:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ContractError("FrameId must be an integer")
        if not 0 <= frame_id < self.frame_count:
            raise ContractError("FrameId {} is outside this sequence".format(frame_id))

    def read_stereo(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_frame_id(frame_id)
        left = cv2.imread(str(self.left_images[frame_id]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(self.right_images[frame_id]), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise ContractError("cannot decode stereo images for FrameId {}".format(frame_id))
        return left, right


def _discover_kitti_sequence(
    dataset_root: Path,
    sequence_id: str,
    *,
    include_ground_truth: bool,
    ground_truth_convention: str = "world-to-camera",
) -> KittiStereoSequence:
    root = Path(dataset_root).expanduser().resolve()
    normalized = normalize_sequence_id(sequence_id)
    selected_ground_truth_convention = str(ground_truth_convention)
    if include_ground_truth:
        from .ground_truth import normalize_ground_truth_convention

        selected_ground_truth_convention = normalize_ground_truth_convention(
            ground_truth_convention
        )
    sequence_root = root / "sequences" / normalized
    left_by_id, ignored_left = _discover_image_directory(sequence_root / "image_0")
    right_by_id, ignored_right = _discover_image_directory(sequence_root / "image_1")
    if set(left_by_id) != set(right_by_id):
        left_only = sorted(set(left_by_id) - set(right_by_id))
        right_only = sorted(set(right_by_id) - set(left_by_id))
        raise ContractError(
            "left/right canonical FrameIds differ; left-only={}, right-only={}".format(
                left_only[:10], right_only[:10]
            )
        )
    frame_ids = tuple(sorted(left_by_id))
    if not frame_ids:
        raise ContractError("KITTI sequence contains no canonical stereo frames")
    if frame_ids != tuple(range(frame_ids[-1] + 1)):
        raise ContractError("KITTI FrameIds must be contiguous and start at zero")
    if len(frame_ids) < 5:
        raise ContractError("validated Bundle policy requires at least five frames")
    calibration_path = sequence_root / "calib.txt"
    calibration = load_stereo_calibration(calibration_path)
    ground_truth_path = root / "poses" / (normalized + ".txt")
    ground_truth = None
    selected_ground_truth_path: Optional[Path] = None
    if include_ground_truth and ground_truth_path.is_file():
        from .ground_truth import load_ground_truth_world_to_camera

        ground_truth = load_ground_truth_world_to_camera(
            ground_truth_path,
            len(frame_ids),
            selected_ground_truth_convention,
        )
        selected_ground_truth_path = ground_truth_path
    identity = {
        "sequence_id": normalized,
        "dataset_root": str(root),
        "frame_ids": list(frame_ids),
        "calibration": calibration.record(),
    }
    dataset_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return KittiStereoSequence(
        dataset_root=root,
        sequence_id=normalized,
        sequence_root=sequence_root,
        frame_ids=frame_ids,
        left_images=tuple(left_by_id[index] for index in frame_ids),
        right_images=tuple(right_by_id[index] for index in frame_ids),
        ignored_left_entries=ignored_left,
        ignored_right_entries=ignored_right,
        calibration_path=calibration_path,
        calibration=calibration,
        ground_truth_path=selected_ground_truth_path,
        ground_truth_world_to_camera=ground_truth,
        ground_truth_input_convention=selected_ground_truth_convention,
        dataset_id=dataset_id,
    )


def discover_kitti_sequence(
    dataset_root: Path,
    sequence_id: str,
    *,
    ground_truth_convention: str = "world-to-camera",
) -> KittiStereoSequence:
    """Discover the complete dataset plus optional analysis-only Ground Truth."""

    return _discover_kitti_sequence(
        dataset_root,
        sequence_id,
        include_ground_truth=True,
        ground_truth_convention=ground_truth_convention,
    )


def discover_kitti_estimation_context(
    dataset_root: Path, sequence_id: str
) -> KittiEstimationContext:
    """Discover only inputs visible to estimation-stage child processes."""

    return _discover_kitti_sequence(
        dataset_root, sequence_id, include_ground_truth=False
    ).estimation_context()


__all__ = [
    "CANONICAL_IMAGE_NAME",
    "KittiStereoSequence",
    "KittiEstimationContext",
    "discover_kitti_estimation_context",
    "discover_kitti_sequence",
    "normalize_sequence_id",
]
