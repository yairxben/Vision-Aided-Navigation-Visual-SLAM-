"""Tracking transitions computed by the C++ module ``generic_slam._native``.

The native kernels reproduce the Python frontend bit for bit (see
native/src/module.cpp); this module adapts them to the tracking stage. Small
per-frame matrix products stay in numpy here, exactly as in the Python path.
"""

from __future__ import annotations

import random
from typing import Mapping

import numpy as np

from ..contracts import ContractError
from ..frontend.pnp import CameraModel, required_hypothesis_count
from ..frontend.stereo import compose_right_camera
from .adapter import MatchArrays, PreparedStereoFrame
from .store import NO_ID

try:
    from .. import _native
except ImportError as error:  # pragma: no cover - depends on the local build
    _native = None
    _IMPORT_ERROR = error


def _module():
    if _native is None:
        raise ContractError(
            "frontend.backend 'native' needs generic_slam._native; build it with "
            "tools/build_native.sh ({})".format(_IMPORT_ERROR)
        )
    return _native


class NativeFrontend:
    """Stereo extraction for the native backend (AKAZE, global nearest match)."""

    # Brute-force Hamming matching has no random state.
    is_deterministic = True

    def __init__(self, frontend_config: Mapping[str, object]) -> None:
        stereo = frontend_config["stereo"]
        self._threshold = float(frontend_config["akaze"]["threshold"])
        self._maximum_vertical_deviation_px = float(stereo["maximum_vertical_deviation_px"])
        self._require_positive_disparity = bool(stereo["require_positive_disparity"])
        self._native = _module()

    def extract_stereo(self, left_image: np.ndarray, right_image: np.ndarray):
        for image, label in ((left_image, "left"), (right_image, "right")):
            if image is None or np.asarray(image).ndim != 2:
                raise ContractError("{} stereo image must be a loaded grayscale image".format(label))
        try:
            return self._native.extract_stereo(
                left_image, right_image, self._threshold,
                self._maximum_vertical_deviation_px, self._require_positive_disparity,
            )
        except self._native.ContractViolation as error:
            raise ContractError(str(error)) from error


def prepare_native_frame(frame) -> PreparedStereoFrame:
    """PreparedStereoFrame of a native frame (stereo inliers are one per left feature)."""

    links = frame.links()
    if not len(links):
        raise ContractError("the tracking store cannot accept an empty stereo population")
    filtered_to_raw = frame.stereo_rows()
    raw_to_filtered = np.full(frame.left_feature_count, NO_ID, dtype=np.int64)
    raw_to_filtered[filtered_to_raw] = np.arange(len(filtered_to_raw))
    return PreparedStereoFrame(
        raw_descriptor_count=int(frame.left_feature_count),
        links=links,
        raw_to_filtered=raw_to_filtered,
        filtered_to_raw=filtered_to_raw,
    )


def native_transition(
    previous,
    current,
    previous_pose: np.ndarray,
    intrinsic: np.ndarray,
    fixed_right: np.ndarray,
    vo_ransac: Mapping[str, object],
):
    """One VO transition; returns the C++ result dict plus the four-camera model.

    Consumes Python's ``random`` state exactly as the Python RANSAC would.
    """

    native = _module()
    intrinsic = np.asarray(intrinsic, dtype=float)
    previous_rotation = previous_pose[:3, :3]
    previous_translation = previous_pose[:3, 3:]
    fixed_rotation = fixed_right[:3, :3]
    fixed_translation = fixed_right[:3, 3:]
    previous_right_rotation, previous_right_translation = compose_right_camera(
        previous_rotation, previous_translation, fixed_rotation, fixed_translation,
    )
    # triangulate_stereo_matches' projection matrices, computed the same way.
    projection_left = intrinsic @ np.hstack((
        np.asarray(previous_rotation, dtype=float),
        np.asarray(previous_translation, dtype=float).reshape(3, 1),
    ))
    projection_right = intrinsic @ np.hstack((
        np.asarray(previous_right_rotation, dtype=float),
        np.asarray(previous_right_translation, dtype=float).reshape(3, 1),
    ))
    success_probability = float(vo_ransac["success_probability"])
    sample_size = int(vo_ransac["sample_size"])
    maximum_attempts = vo_ransac["maximum_hypothesis_attempts"]
    version, internal, gauss = random.getstate()
    try:
        result = native.track_transition(
            previous, current, projection_left, projection_right, intrinsic,
            previous_rotation, previous_translation,
            previous_right_rotation, previous_right_translation,
            fixed_rotation, fixed_translation,
            success_probability,
            required_hypothesis_count(
                success_probability, float(vo_ransac["initial_outlier_probability"]), sample_size
            ),
            sample_size,
            float(vo_ransac["supporter_max_distance_px"]),
            None if maximum_attempts is None else int(maximum_attempts),
            required_hypothesis_count,
            list(internal),
        )
    except native.ContractViolation as error:
        raise ContractError(str(error)) from error
    random.setstate((version, tuple(result["random_state"]), gauss))
    rotation, translation = result["rotation"], result["translation"]
    current_right_rotation, current_right_translation = compose_right_camera(
        rotation, translation, fixed_rotation, fixed_translation,
    )
    model = CameraModel(
        rotations=(
            np.asarray(previous_rotation, dtype=float), previous_right_rotation,
            rotation, current_right_rotation,
        ),
        translations=(
            np.asarray(previous_translation, dtype=float).reshape(3, 1),
            previous_right_translation, translation, current_right_translation,
        ),
    )
    reciprocal = MatchArrays(
        result["reciprocal_query"], result["reciprocal_train"], result["reciprocal_distance"]
    )
    return result, model, reciprocal


__all__ = ["NativeFrontend", "native_transition", "prepare_native_frame"]
