"""Configurable AKAZE/SIFT extraction and descriptor correspondence policy."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import BINARY_DETECTORS
from ..contracts import ContractError

# One helper thread detects the right image while the caller detects the left
# (OpenCV releases the GIL): measured 114 -> 94 ms per stereo pair on KITTI.
_SIDE_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _side_executor() -> ThreadPoolExecutor:
    global _SIDE_EXECUTOR
    if _SIDE_EXECUTOR is None:
        _SIDE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frontend-side")
    return _SIDE_EXECUTOR


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
    if detector == "orb":
        orb = frontend_config["orb"]
        return cv2.ORB_create(
            nfeatures=int(orb["max_features"]),
            scaleFactor=float(orb["scale_factor"]),
            nlevels=int(orb["levels"]),
            fastThreshold=int(orb["fast_threshold"]),
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
        # A second instance: OpenCV detector objects are not shared across threads.
        self._right_detector = create_detector(frontend_config)
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
        self.stereo_matching = str(stereo["matching"])
        self.row_band_uniqueness = str(stereo["row_band_uniqueness"])
        self.row_band_ratio = float(stereo["row_band_ratio"])

    def create_matcher(self) -> cv2.DescriptorMatcher:
        """Create an independent matcher whose norm fits the descriptor type.

        AKAZE descriptors are binary, so they use exact Hamming brute force
        (a FLANN KD-tree index only supports float descriptors). SIFT keeps
        the configured FLANN KD-tree with the L2 norm.
        """

        if self.detector_type in BINARY_DETECTORS:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return cv2.FlannBasedMatcher(
            dict(self._index_parameters), dict(self._search_parameters)
        )

    @property
    def is_deterministic(self) -> bool:
        """True when matching uses no random state (exact brute force).

        FLANN's randomized KD-trees draw from OpenCV's RNG, so for SIFT the
        order of matcher calls matters and they must stay sequential.
        """

        return self.detector_type in BINARY_DETECTORS

    @staticmethod
    def _extract_with(detector: cv2.Feature2D, image: np.ndarray, label: str) -> ImageFeatures:
        if image is None or np.asarray(image).ndim != 2:
            raise ContractError("{} must be a loaded grayscale image".format(label))
        keypoints, descriptors = detector.detectAndCompute(image, None)
        return ImageFeatures(
            keypoints=tuple(keypoints),
            descriptors=_require_descriptors(descriptors, label),
        )

    def extract(self, image: np.ndarray, label: str = "image") -> ImageFeatures:
        return self._extract_with(self._detector, image, label)

    def one_way_matches(
        self, descriptors_query: np.ndarray, descriptors_train: np.ndarray
    ) -> Tuple[cv2.DMatch, ...]:
        matches = self.create_matcher().match(descriptors_query, descriptors_train)
        return tuple(matches)

    def extract_stereo(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> StereoFeatures:
        # Detection has no random state, so the two images run in parallel.
        right_future = _side_executor().submit(
            self._extract_with, self._right_detector, right_image, "right stereo image"
        )
        left = self.extract(left_image, "left stereo image")
        right = right_future.result()
        if self.stereo_matching == "row_band":
            return StereoFeatures(
                left=left, right=right, inliers=self._row_band_matches(left, right),
                outliers=(),
            )
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

    # Left features are processed in horizontal stripes of this height; each
    # stripe is compared only with right features within the vertical tolerance.
    ROW_BAND_STRIPE_PX = 8.0

    def _row_band_candidates(
        self, left: ImageFeatures, right: ImageFeatures
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every (left, right, distance) pair that passes the stereo geometry."""

        left_points = np.array([keypoint.pt for keypoint in left.keypoints], dtype=float)
        right_points = np.array([keypoint.pt for keypoint in right.keypoints], dtype=float)
        tolerance = self.maximum_vertical_deviation_px
        right_order = np.argsort(right_points[:, 1], kind="stable")
        right_y = right_points[right_order, 1]
        stripes = np.floor(left_points[:, 1] / self.ROW_BAND_STRIPE_PX).astype(np.int64)
        binary = self.detector_type in BINARY_DETECTORS
        norm, dtype = (cv2.NORM_HAMMING, cv2.CV_32S) if binary else (cv2.NORM_L2, cv2.CV_32F)
        queries, trains, distances = [], [], []
        for stripe in np.unique(stripes):
            left_rows = np.nonzero(stripes == stripe)[0]
            low = np.searchsorted(right_y, stripe * self.ROW_BAND_STRIPE_PX - tolerance, "left")
            high = np.searchsorted(
                right_y, (stripe + 1) * self.ROW_BAND_STRIPE_PX + tolerance, "right"
            )
            right_rows = right_order[low:high]
            if len(right_rows) == 0:
                continue
            sorted_distance, sorted_index = cv2.batchDistance(
                left.descriptors[left_rows], right.descriptors[right_rows], dtype,
                normType=norm, K=len(right_rows),
            )
            distance = np.empty(sorted_distance.shape, dtype=float)
            np.put_along_axis(distance, sorted_index.astype(np.int64), sorted_distance, axis=1)
            dy = np.abs(left_points[left_rows, 1][:, None] - right_points[right_rows, 1][None, :])
            valid = dy <= tolerance
            if self.require_positive_disparity:
                valid &= left_points[left_rows, 0][:, None] - right_points[right_rows, 0][None, :] > 0.0
            rows, columns = np.nonzero(valid)
            queries.append(left_rows[rows])
            trains.append(right_rows[columns])
            distances.append(distance[rows, columns])
        if not queries:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0, dtype=float)
        return np.concatenate(queries), np.concatenate(trains), np.concatenate(distances)

    def _row_band_matches(
        self, left: ImageFeatures, right: ImageFeatures
    ) -> Tuple[cv2.DMatch, ...]:
        """Nearest geometrically valid right feature per left feature, made unique.

        Ties break toward the lower feature index. "mutual" keeps a pair only
        if the left feature is also the right feature's nearest valid partner;
        "ratio" keeps it when best < ratio * second best (or it is the only one).
        """

        query, train, distance = self._row_band_candidates(left, right)
        if len(query) == 0:
            return ()
        by_left = np.lexsort((train, distance, query))
        first = np.ones(len(by_left), dtype=bool)
        first[1:] = query[by_left][1:] != query[by_left][:-1]
        best = by_left[first]  # one candidate per left feature, ascending query
        if self.row_band_uniqueness == "mutual":
            by_right = np.lexsort((query, distance, train))
            first_right = np.ones(len(by_right), dtype=bool)
            first_right[1:] = train[by_right][1:] != train[by_right][:-1]
            best_left_of_right = dict(
                zip(train[by_right][first_right].tolist(), query[by_right][first_right].tolist())
            )
            keep = np.array(
                [best_left_of_right[t] == q for q, t in zip(query[best].tolist(), train[best].tolist())],
                dtype=bool,
            )
        else:
            group_start = np.flatnonzero(first)
            group_size = np.diff(np.append(group_start, len(by_left)))
            second = np.full(len(best), np.inf)
            has_second = group_size > 1
            second[has_second] = distance[by_left[group_start[has_second] + 1]]
            keep = distance[best] < self.row_band_ratio * second
        chosen = best[keep]
        return tuple(
            cv2.DMatch(int(q), int(t), 0, float(d))
            for q, t, d in zip(query[chosen], train[chosen], distance[chosen])
        )

    def reciprocal_matches(
        self, descriptors_back: np.ndarray, descriptors_front: np.ndarray
    ) -> Tuple[cv2.DMatch, ...]:
        """Retain forward matches whose reverse nearest match is identical."""

        # Kept sequential on purpose: OpenCV's matcher already uses every core,
        # and two concurrent matches measured 2x slower than one after another.
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


    def reciprocal_matches_on_rows(
        self,
        descriptors_back: np.ndarray,
        back_rows: Sequence[int],
        descriptors_front: np.ndarray,
        front_rows: Sequence[int],
    ) -> Tuple[cv2.DMatch, ...]:
        """Reciprocal matches restricted to the given ascending descriptor rows.

        Indices in the result refer to the full descriptor arrays, so callers
        see the same index space as :meth:`reciprocal_matches`.
        """

        back_rows = np.asarray(back_rows, dtype=int)
        front_rows = np.asarray(front_rows, dtype=int)
        if np.any(np.diff(back_rows) <= 0) or np.any(np.diff(front_rows) <= 0):
            raise ContractError("matching rows must be strictly ascending")
        if len(back_rows) == 0 or len(front_rows) == 0:
            return ()
        local = self.reciprocal_matches(
            descriptors_back[back_rows], descriptors_front[front_rows]
        )
        return tuple(
            cv2.DMatch(
                int(back_rows[match.queryIdx]), int(front_rows[match.trainIdx]),
                int(match.imgIdx), float(match.distance),
            )
            for match in local
        )


def stereo_rows(stereo: StereoFeatures) -> Tuple[int, ...]:
    """Ascending left-feature indices that have a stereo inlier."""

    return tuple(sorted({int(match.queryIdx) for match in stereo.inliers}))


def match_triplets(matches: Sequence[cv2.DMatch]) -> Tuple[Tuple[int, int, float], ...]:
    """Stable primitive representation used by bounded parity tests/artifacts."""

    return tuple(
        (int(match.queryIdx), int(match.trainIdx), float(match.distance))
        for match in matches
    )
