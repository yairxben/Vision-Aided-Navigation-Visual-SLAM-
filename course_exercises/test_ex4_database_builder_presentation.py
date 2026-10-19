"""Focused E4-R02 scientific-builder presentation tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import exercise4_database_builder as builder
from tracking_database import NO_ID
from tracking_database_adapter import (
    PreparedStereoFrame,
    PreparedTemporalTransition,
)


def _raw_frame(frame_id: int) -> builder.RawStereoFrame:
    keypoints = (
        cv2.KeyPoint(1.0, 2.0, 1.0),
        cv2.KeyPoint(3.0, 4.0, 1.0),
    )
    stereo_inliers = (
        cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.1),
        cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=0.2),
    )
    return builder.RawStereoFrame(
        frame_id=frame_id,
        keypoints_left=keypoints,
        descriptors_left=np.full((2, 4), frame_id + 1, dtype=np.float32),
        keypoints_right=keypoints,
        right_descriptor_count=2,
        raw_stereo_match_count=3,
        stereo_inliers=stereo_inliers,
    )


def _prepared_frame() -> PreparedStereoFrame:
    return PreparedStereoFrame(
        raw_descriptor_count=2,
        filtered_descriptors=np.ones((2, 4), dtype=np.float32),
        links=[],
        stereo_matches=[],
        raw_to_filtered=(0, 1),
        filtered_to_raw=(0, 1),
    )


def _prepared_transition() -> PreparedTemporalTransition:
    return PreparedTemporalTransition(
        matches=[
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.1),
            cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=0.2),
        ],
        validity_mask=[True, True],
        reciprocal_candidate_count=3,
        stereo_compatible_candidate_count=2,
        selected_candidate_count=2,
        consensus_count=2,
        supporter_count=2,
        selected_supporter_count=2,
        supporter_percentage=100.0,
        selected_candidates=[],
        rejected_candidates=[],
    )


class _CountingMatcherFixture:
    def __init__(self):
        self.calls = 0

    def match(self, *_args, **_kwargs):
        self.calls += 1
        count = 3 if self.calls == 1 else 2
        return [
            cv2.DMatch(
                _queryIdx=index,
                _trainIdx=index,
                _distance=float(index),
            )
            for index in range(count)
        ]


class EstimatedTransitionTest(unittest.TestCase):
    def test_stage_order_counts_and_counter_restoration(self):
        order = []
        raw_back = _raw_frame(0)
        raw_front = _raw_frame(1)
        prepared_front = _prepared_frame()
        reciprocal = [
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.1)
        ]
        consensus = [(0, 0, 0)]
        supporters = [(0, 0, 0)]
        matcher = _CountingMatcherFixture()
        front_rotation = np.eye(3)
        front_translation = np.asarray([[1.0], [2.0], [3.0]])
        pnp_results = [
            (True, np.eye(3), np.zeros((3, 1))),
            (False, np.eye(3), np.zeros((3, 1))),
        ]

        def right_camera(*_args, **_kwargs):
            order.append("right-camera")
            return np.eye(3), np.zeros((3, 1))

        def triangulate(*_args, **_kwargs):
            order.append("triangulate")
            return np.zeros((2, 3))

        def extract(frame_id):
            order.append("extract-front")
            self.assertEqual(frame_id, 1)
            return raw_front, prepared_front

        def reciprocal_match(*_args, matcher, **_kwargs):
            order.append("reciprocal-match")
            matcher.match(None, None)
            matcher.match(None, None)
            return reciprocal

        def consensus_match(*_args, **_kwargs):
            order.append("consensus")
            return consensus

        def original_build_model(*_args, **_kwargs):
            return None

        def original_front_camera(*_args, **_kwargs):
            return pnp_results.pop(0)

        def ransac(*_args, **_kwargs):
            order.append("ransac")
            builder.tools.build_model(use_random=True)
            builder.tools.build_model(use_random=False)
            builder.tools.calculate_front_camera_matrix()
            builder.tools.calculate_front_camera_matrix()
            rotations = [np.eye(3), np.eye(3), front_rotation, np.eye(3)]
            translations = [
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                front_translation,
                np.zeros((3, 1)),
            ]
            return rotations, translations, supporters, [0]

        with mock.patch.object(
            builder.tools, "calculate_right_camera_matrix", right_camera
        ), mock.patch.object(
            builder.tools, "cv_triangulate_matched_points", triangulate
        ), mock.patch.object(
            builder, "extract_canonical_stereo_frame", extract
        ), mock.patch.object(
            builder.tools, "MATCHER", matcher
        ), mock.patch.object(
            builder.tools, "match_reciprocal", reciprocal_match
        ), mock.patch.object(
            builder.tools, "find_consensus_matches_indices", consensus_match
        ), mock.patch.object(
            builder.tools, "build_model", original_build_model
        ), mock.patch.object(
            builder.tools,
            "calculate_front_camera_matrix",
            original_front_camera,
        ), mock.patch.object(
            builder.tools, "estimate_projection_matrices_with_ransac", ransac
        ):
            result = builder._estimate_consecutive_frame_transition(
                raw_back=raw_back,
                front_frame_id=1,
                back_world_to_camera_pose=np.hstack(
                    (np.eye(3), np.zeros((3, 1)))
                ),
                intrinsic=np.eye(3),
                initial_right_rotation=np.eye(3),
                initial_right_translation=np.zeros((3, 1)),
            )
            self.assertIs(builder.tools.build_model, original_build_model)
            self.assertIs(
                builder.tools.calculate_front_camera_matrix,
                original_front_camera,
            )

        self.assertEqual(
            order[:6],
            [
                "right-camera",
                "triangulate",
                "extract-front",
                "reciprocal-match",
                "consensus",
                "ransac",
            ],
        )
        self.assertIs(result.front_raw_frame, raw_front)
        self.assertIs(result.front_prepared_frame, prepared_front)
        self.assertIs(result.reciprocal_matches, reciprocal)
        self.assertIs(result.consensus, consensus)
        self.assertIs(result.final_supporters, supporters)
        np.testing.assert_array_equal(
            result.front_world_to_camera_pose,
            np.hstack((front_rotation, front_translation)),
        )
        self.assertEqual(result.temporal_forward_match_count, 3)
        self.assertEqual(result.temporal_reverse_match_count, 2)
        self.assertEqual(result.ransac_random_hypothesis_count, 1)
        self.assertEqual(result.ransac_refinement_model_count, 1)
        self.assertEqual(result.pnp_attempt_count, 2)
        self.assertEqual(result.pnp_failure_count, 1)

    def test_ransac_counter_monkey_patches_restore_on_failure(self):
        raw_back = _raw_frame(0)
        raw_front = _raw_frame(1)
        matcher = _CountingMatcherFixture()

        def reciprocal_match(*_args, matcher, **_kwargs):
            matcher.match(None, None)
            matcher.match(None, None)
            return []

        def original_build_model(*_args, **_kwargs):
            return None

        def original_front_camera(*_args, **_kwargs):
            return True, np.eye(3), np.zeros((3, 1))

        def fail_ransac(*_args, **_kwargs):
            raise RuntimeError("bounded RANSAC failure")

        with mock.patch.object(
            builder.tools,
            "calculate_right_camera_matrix",
            return_value=(np.eye(3), np.zeros((3, 1))),
        ), mock.patch.object(
            builder.tools,
            "cv_triangulate_matched_points",
            return_value=np.zeros((2, 3)),
        ), mock.patch.object(
            builder,
            "extract_canonical_stereo_frame",
            return_value=(raw_front, _prepared_frame()),
        ), mock.patch.object(
            builder.tools, "MATCHER", matcher
        ), mock.patch.object(
            builder.tools, "match_reciprocal", reciprocal_match
        ), mock.patch.object(
            builder.tools,
            "find_consensus_matches_indices",
            return_value=[],
        ), mock.patch.object(
            builder.tools, "build_model", original_build_model
        ), mock.patch.object(
            builder.tools,
            "calculate_front_camera_matrix",
            original_front_camera,
        ), mock.patch.object(
            builder.tools,
            "estimate_projection_matrices_with_ransac",
            fail_ransac,
        ):
            with self.assertRaisesRegex(RuntimeError, "RANSAC failure"):
                builder._estimate_consecutive_frame_transition(
                    raw_back=raw_back,
                    front_frame_id=1,
                    back_world_to_camera_pose=np.hstack(
                        (np.eye(3), np.zeros((3, 1)))
                    ),
                    intrinsic=np.eye(3),
                    initial_right_rotation=np.eye(3),
                    initial_right_translation=np.zeros((3, 1)),
                )
            self.assertIs(builder.tools.build_model, original_build_model)
            self.assertIs(
                builder.tools.calculate_front_camera_matrix,
                original_front_camera,
            )


class InsertedTransitionTest(unittest.TestCase):
    def test_final_supporter_insertion_and_lifecycle_evidence(self):
        transition = _prepared_transition()
        database = SimpleNamespace(
            frameId_to_trackIds_list={0: [10, 11, NO_ID]},
            last_trackId=11,
        )
        database.tracks = mock.Mock(return_value=[10, 12])

        def add_frame(*_args):
            database.last_trackId = 12
            return 1, transition

        adapter = SimpleNamespace(
            database=database,
            add_frame=mock.Mock(side_effect=add_frame),
        )
        estimated = builder._EstimatedConsecutiveTransition(
            front_raw_frame=_raw_frame(1),
            front_prepared_frame=_prepared_frame(),
            reciprocal_matches=[object(), object(), object()],
            consensus=[(0, 0, 0), (1, 1, 1)],
            final_supporters=[(0, 0, 0), (1, 1, 1)],
            front_world_to_camera_pose=np.hstack(
                (np.eye(3), np.zeros((3, 1)))
            ),
            temporal_forward_match_count=3,
            temporal_reverse_match_count=3,
            ransac_random_hypothesis_count=1,
            ransac_refinement_model_count=1,
            pnp_attempt_count=1,
            pnp_failure_count=0,
        )
        supporter_only = {
            "connecting_track_count": 2,
            "true_validity_mask_count": 2,
            "final_supporter_validity_exact": True,
            "selected_consensus_non_supporter_count": 0,
        }
        rejections = {"total_preflight_candidate_rejections": 1}
        consistency_frame_ids = [0]

        with mock.patch.object(
            builder, "validate_tracking_database_consistency_linear"
        ) as validate, mock.patch.object(
            builder,
            "_verify_supporter_only_insertion",
            return_value=supporter_only,
        ), mock.patch.object(
            builder, "_rejection_counts", return_value=rejections
        ):
            result = builder._insert_supported_transition(
                adapter=adapter,
                back_frame_id=0,
                front_frame_id=1,
                estimated=estimated,
                validate_after_insertion=True,
                consistency_frame_ids=consistency_frame_ids,
            )

        validate.assert_called_once_with(database)
        adapter.add_frame.assert_called_once_with(
            estimated.front_prepared_frame,
            estimated.reciprocal_matches,
            estimated.final_supporters,
            len(estimated.consensus),
            estimated.front_world_to_camera_pose,
        )
        self.assertEqual(consistency_frame_ids, [0, 1])
        self.assertEqual(result.issued_frame_id, 1)
        self.assertIs(result.prepared_transition, transition)
        self.assertEqual(result.tracks_created, 1)
        self.assertEqual(result.tracks_extended, 1)
        self.assertEqual(result.tracks_terminated, 1)
        self.assertIs(result.supporter_only_validation, supporter_only)
        self.assertIs(result.rejection_counts, rejections)


class TransitionMetadataTest(unittest.TestCase):
    def test_field_order_and_runtime_field_are_preserved(self):
        estimated = builder._EstimatedConsecutiveTransition(
            front_raw_frame=_raw_frame(1),
            front_prepared_frame=_prepared_frame(),
            reciprocal_matches=[object(), object(), object()],
            consensus=[(0, 0, 0), (1, 1, 1)],
            final_supporters=[(0, 0, 0)],
            front_world_to_camera_pose=np.hstack(
                (np.eye(3), np.zeros((3, 1)))
            ),
            temporal_forward_match_count=4,
            temporal_reverse_match_count=5,
            ransac_random_hypothesis_count=6,
            ransac_refinement_model_count=7,
            pnp_attempt_count=8,
            pnp_failure_count=9,
        )
        inserted = builder._InsertedTransitionEvidence(
            issued_frame_id=1,
            prepared_transition=_prepared_transition(),
            tracks_created=10,
            tracks_extended=11,
            tracks_terminated=12,
            supporter_only_validation={
                "connecting_track_count": 13,
                "true_validity_mask_count": 14,
                "final_supporter_validity_exact": True,
                "selected_consensus_non_supporter_count": 15,
            },
            rejection_counts={"total_preflight_candidate_rejections": 1},
        )
        expected_fields = [
            "back_frame",
            "front_frame",
            "success",
            "failure",
            "back_left_feature_count",
            "front_left_feature_count",
            "back_right_feature_count",
            "front_right_feature_count",
            "back_stereo_row_count",
            "front_stereo_row_count",
            "stereo_raw_match_count_back",
            "stereo_raw_match_count_front",
            "temporal_forward_match_count",
            "temporal_reverse_match_count",
            "reciprocal_match_count",
            "consensus_count",
            "supporter_count",
            "supporter_percentage",
            "complete_temporal_row_count",
            "tracks_created",
            "tracks_extended",
            "tracks_terminated",
            "connectivity",
            "true_validity_mask_count",
            "final_supporter_validity_exact",
            "selected_consensus_non_supporter_count",
            "non_supporters_entered",
            "ransac_random_hypothesis_count",
            "ransac_refinement_model_count",
            "pnp_attempt_count",
            "pnp_failure_count",
            "preflight_rejections",
            "runtime_seconds",
        ]

        with mock.patch.object(builder.time, "perf_counter", return_value=15.0):
            metadata = builder._build_transition_metadata(
                back_frame_id=0,
                front_frame_id=1,
                raw_back=_raw_frame(0),
                estimated=estimated,
                inserted=inserted,
                back_stereo_row_count=16,
                front_stereo_row_count=17,
                stereo_raw_match_count_back=18,
                stereo_raw_match_count_front=19,
                transition_start=10.0,
            )

        self.assertEqual(list(metadata), expected_fields)
        self.assertEqual(metadata["runtime_seconds"], 5.0)
        self.assertEqual(metadata["temporal_forward_match_count"], 4)
        self.assertEqual(metadata["ransac_random_hypothesis_count"], 6)
        self.assertEqual(metadata["tracks_created"], 10)
        self.assertIs(metadata["preflight_rejections"], inserted.rejection_counts)


if __name__ == "__main__":
    unittest.main()
