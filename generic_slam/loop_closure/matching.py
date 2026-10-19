"""Independent arbitrary-pair four-view matching for Loop Closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AlgorithmConfig
from ..dataset.kitti import KittiEstimationContext
from ..frontend.features import FeatureFrontend, StereoFeatures
from ..frontend.pnp import BoundedRansacResult, reset_random_generators, run_bounded_four_view_ransac
from ..frontend.stereo import compose_right_camera, triangulate_stereo_matches
from ..frontend.temporal import FourViewCorrespondence, build_four_view_consensus


@dataclass(frozen=True)
class LoopMatchResult:
    previous_frame_id: int
    current_frame_id: int
    previous_stereo: StereoFeatures
    current_stereo: StereoFeatures
    reciprocal_matches: tuple
    consensus: tuple
    previous_points_3d: np.ndarray
    ransac: BoundedRansacResult

    @property
    def supporter_count(self) -> int:
        return len(self.ransac.best_supporter_indices)

    @property
    def supporter_percentage(self) -> float:
        return (
            100.0 * self.supporter_count / len(self.consensus)
            if self.consensus else 0.0
        )


def evaluate_loop_pair(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    previous_frame_id: int,
    current_frame_id: int,
) -> LoopMatchResult:
    loop = config.section("loop_closure")
    ransac_config = loop["ransac"]
    reset_random_generators(int(loop["pair_seed"]))
    frontend = FeatureFrontend(config.section("frontend"))
    previous_images = dataset.read_stereo(previous_frame_id)
    current_images = dataset.read_stereo(current_frame_id)
    previous_stereo = frontend.extract_stereo(*previous_images)
    current_stereo = frontend.extract_stereo(*current_images)
    reciprocal = frontend.reciprocal_matches(
        previous_stereo.left.descriptors, current_stereo.left.descriptors
    )
    consensus = build_four_view_consensus(
        previous_stereo.inliers, current_stereo.inliers, reciprocal
    )
    calibration = dataset.calibration
    left = calibration.left_extrinsic
    right = calibration.right_from_left
    previous_right_rotation, previous_right_translation = compose_right_camera(
        left[:3, :3], left[:3, 3:], right[:3, :3], right[:3, 3:]
    )
    points = triangulate_stereo_matches(
        previous_stereo.left.keypoints,
        previous_stereo.right.keypoints,
        previous_stereo.inliers,
        calibration.intrinsic,
        left[:3, :3], left[:3, 3:],
        previous_right_rotation, previous_right_translation,
    )
    ransac = run_bounded_four_view_ransac(
        points, consensus, previous_stereo.inliers, current_stereo.inliers,
        previous_stereo.left.keypoints, previous_stereo.right.keypoints,
        current_stereo.left.keypoints, current_stereo.right.keypoints,
        calibration.intrinsic, left[:3, :3], left[:3, 3:],
        right[:3, :3], right[:3, 3:],
        success_probability=float(ransac_config["success_probability"]),
        initial_outlier_probability=float(ransac_config["initial_outlier_probability"]),
        sample_size=int(ransac_config["sample_size"]),
        maximum_distance_px=float(ransac_config["supporter_max_distance_px"]),
        maximum_attempts=int(ransac_config["maximum_hypothesis_attempts"]),
    )
    return LoopMatchResult(
        int(previous_frame_id), int(current_frame_id), previous_stereo,
        current_stereo, reciprocal, consensus, points, ransac,
    )


__all__ = ["LoopMatchResult", "evaluate_loop_pair"]
