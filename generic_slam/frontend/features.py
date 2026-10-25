"""Configurable AKAZE/SIFT extraction and descriptor correspondence policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..contracts import ContractError


@dataclass(frozen=True)
class ImageFeatures:
    """One image's keypoints and descriptors."""

    keypoints: Tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray


@dataclass(frozen=True)
class StereoFeatures:
    """Feature and one-nearest-neighbour match evidence for a stereo frame."""

    left: ImageFeatures
    right: ImageFeatures
    inliers: Tuple[cv2.DMatch, ...]
    outliers: Tuple[cv2.DMatch, ...]

    @property
    def raw_match_count(self) -> int:
        return len(self.inliers) + len(self.outliers)


def _require_descriptors(descriptors: Optional[np.ndarray], label: str) -> np.ndarray:
    if descriptors is None:
        raise ContractError("feature detector produced no descriptors for {}".format(label))
    value = np.asarray(descriptors)
    if value.ndim != 2 or value.shape[0] == 0 or not np.all(np.isfinite(value)):
        raise ContractError("invalid feature descriptors for {}".format(label))
    return value


def create_detector(frontend_config: Mapping[str, object]) -> cv2.Feature2D:
    """Build the configured detector/descriptor extractor."""

    detector = frontend_config["detector"]
    if detector == "akaze":
        return cv2.AKAZE_create(
            threshold=float(frontend_config["akaze"]["threshold"])
        )
    if detector == "sift":
        sift = frontend_config["sift"]
        return cv2.SIFT_create(
            contrastThreshold=float(sift["contrast_threshold"]),
            edgeThreshold=float(sift["edge_threshold"]),
        )
    raise ContractError("unsupported frontend.detector: {}".format(detector))


class FeatureFrontend:
    """Stage-scoped detector and matcher with the configured numerical settings."""

    def __init__(self, frontend_config: Mapping[str, object]) -> None:
        flann = frontend_config["flann"]
        stereo = frontend_config["stereo"]
        self.detector_type = str(frontend_config["detector"])
        self._detector = create_detector(frontend_config)
        self._index_parameters = {
            "algorithm": int(flann["algorithm"]),
            "trees": int(flann["trees"]),
        }
        self._search_parameters = {"checks": int(flann["checks"])}
        self.maximum_vertical_deviation_px = float(
            stereo["maximum_vertical_deviation_px"]
        )
        self.require_positive_disparity = bool(
            stereo["require_positive_disparity"]
        )

    def create_matcher(self) -> cv2.DescriptorMatcher:
        """Create an independent matcher whose norm fits the descriptor type.

        AKAZE descriptors are binary, so they use exact Hamming brute force
        (a FLANN KD-tree index only supports float descriptors). SIFT keeps
        the configured FLANN KD-tree with the L2 norm.
        """

        if self.detector_type == "akaze":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return cv2.FlannBasedMatcher(
            dict(self._index_parameters), dict(self._search_parameters)
        )

    def extract(self, image: np.ndarray, label: str = "image") -> ImageFeatures:
        if image is None or np.asarray(image).ndim != 2:
            raise ContractError("{} must be a loaded grayscale image".format(label))
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        return ImageFeatures(
            keypoints=tuple(keypoints),
            descriptors=_require_descriptors(descriptors, label),
        )

    def one_way_matches(
        self, descriptors_query: np.ndarray, descriptors_train: np.ndarray
    ) -> Tuple[cv2.DMatch, ...]:
        matches = self.create_matcher().match(descriptors_query, descriptors_train)
        return tuple(matches)

    def extract_stereo(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> StereoFeatures:
        left = self.extract(left_image, "left stereo image")
        right = self.extract(right_image, "right stereo image")
        inliers = []
        outliers = []
        for match in self.one_way_matches(left.descriptors, right.descriptors):
            left_point = left.keypoints[match.queryIdx].pt
            right_point = right.keypoints[match.trainIdx].pt
            rectified = (
                abs(left_point[1] - right_point[1])
                <= self.maximum_vertical_deviation_px
            )
            positive_disparity = left_point[0] - right_point[0] > 0.0
            accepted = rectified and (
                positive_disparity or not self.require_positive_disparity
            )
            (inliers if accepted else outliers).append(match)
        return StereoFeatures(
            left=left,
            right=right,
            inliers=tuple(sorted(inliers, key=lambda item: item.queryIdx)),
            outliers=tuple(sorted(outliers, key=lambda item: item.queryIdx)),
        )

    def reciprocal_matches(
        self, descriptors_back: np.ndarray, descriptors_front: np.ndarray
    ) -> Tuple[cv2.DMatch, ...]:
        """Retain forward matches whose reverse nearest match is identical."""

        matcher = self.create_matcher()
        forward = matcher.match(descriptors_back, descriptors_front)
        reverse = matcher.match(descriptors_front, descriptors_back)
        reverse_map = {match.queryIdx: match.trainIdx for match in reverse}
        return tuple(
            sorted(
                (
                    match
                    for match in forward
                    if reverse_map.get(match.trainIdx) == match.queryIdx
                ),
                key=lambda item: item.queryIdx,
            )
        )


def match_triplets(matches: Sequence[cv2.DMatch]) -> Tuple[Tuple[int, int, float], ...]:
    """Stable primitive representation used by bounded parity tests/artifacts."""

    return tuple(
        (int(match.queryIdx), int(match.trainIdx), float(match.distance))
        for match in matches
    )
