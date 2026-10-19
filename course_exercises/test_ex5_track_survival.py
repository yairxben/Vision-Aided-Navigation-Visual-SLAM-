"""Small deterministic contracts for the sole E5-I03C keyframe policy."""

from __future__ import annotations

import copy
import unittest

from exercise5_full_run import (
    Exercise5ContractError,
    load_frozen_configuration,
    validate_frozen_configuration,
)
from exercise5_track_survival import (
    derive_track_survival_keyframes,
    select_track_survival_endpoint,
    validate_track_survival_keyframes,
)


class _SyntheticTrackingDB:
    def __init__(self, last_frames):
        self._last_frames = last_frames
        self._last_by_track = {}

    def tracks(self, frame_id):
        values = tuple(self._last_frames(frame_id))
        track_ids = tuple(frame_id * 100 + index for index in range(len(values)))
        self._last_by_track.update(zip(track_ids, values))
        return list(track_ids)

    def last_frame_of_track(self, track_id):
        return self._last_by_track[track_id]


class TrackSurvivalPolicyTest(unittest.TestCase):
    def test_exact_int_082_selection_rule(self):
        endpoint, index = select_track_survival_endpoint([5, 1, 4, 2, 3, 0])
        self.assertEqual(index, min(5, int(0.82 * 6)))
        self.assertEqual((endpoint, index), (4, 4))
        endpoint, index = select_track_survival_endpoint([4, 0, 3, 1, 2])
        self.assertEqual(index, min(4, int(0.82 * 5)))
        self.assertEqual((endpoint, index), (4, 4))

    def test_minimum_and_maximum_window_constraints(self):
        minimum = derive_track_survival_keyframes(
            _SyntheticTrackingDB(lambda start: [start]),
            final_frame_id=40,
        )
        minimum_lengths = [
            end - start + 1
            for start, end in zip(
                minimum["ordered_keyframe_ids"],
                minimum["ordered_keyframe_ids"][1:],
            )
        ]
        self.assertGreater(
            minimum["statistics"]["selection_adjustment_counts"][
                "minimum_5_frame"
            ],
            0,
        )
        self.assertTrue(all(5 <= length <= 20 for length in minimum_lengths))

        maximum = derive_track_survival_keyframes(
            _SyntheticTrackingDB(lambda _start: [40]),
            final_frame_id=40,
        )
        maximum_lengths = [
            end - start + 1
            for start, end in zip(
                maximum["ordered_keyframe_ids"],
                maximum["ordered_keyframe_ids"][1:],
            )
        ]
        self.assertGreater(
            maximum["statistics"]["selection_adjustment_counts"][
                "maximum_20_frame"
            ],
            0,
        )
        self.assertTrue(all(5 <= length <= 20 for length in maximum_lengths))

    def test_final_tail_repair_is_deterministic(self):
        derived = derive_track_survival_keyframes(
            _SyntheticTrackingDB(lambda _start: [22]),
            final_frame_id=22,
        )
        self.assertEqual(derived["ordered_keyframe_ids"], [0, 18, 22])
        self.assertEqual(
            derived["statistics"]["selection_adjustment_counts"][
                "final_tail_repair"
            ],
            1,
        )

    def test_full_coverage_strict_ordering_and_clear_rejection(self):
        derived = derive_track_survival_keyframes(
            _SyntheticTrackingDB(
                lambda start: [min(53, start + 7)] * 6
            ),
            final_frame_id=53,
        )
        keyframes = validate_track_survival_keyframes(
            derived["ordered_keyframe_ids"], final_frame_id=53
        )
        self.assertEqual(keyframes[0], 0)
        self.assertEqual(keyframes[-1], 53)
        self.assertTrue(
            all(second > first for first, second in zip(keyframes, keyframes[1:]))
        )
        with self.assertRaisesRegex(
            Exercise5ContractError, "must end at 53"
        ):
            validate_track_survival_keyframes(
                keyframes[:-1], final_frame_id=53
            )
        with self.assertRaisesRegex(
            Exercise5ContractError, "5-20 actual-frame"
        ):
            validate_track_survival_keyframes(
                [0, 3, 53], final_frame_id=53
            )

    def test_canonical_e5_i03b_contract_still_rejects_any_keyframe_change(self):
        configuration, _ = load_frozen_configuration()
        changed = copy.deepcopy(configuration)
        changed["keyframe_selection"]["ordered_keyframe_ids"][1] = 8
        with self.assertRaisesRegex(
            Exercise5ContractError, "frozen keyframe contract differs"
        ):
            validate_frozen_configuration(changed)
        validate_frozen_configuration(configuration)


if __name__ == "__main__":
    unittest.main()
