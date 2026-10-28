"""Columnar tracking store: feature tracks over consecutive frames.

Replaces the lecturer's dictionary-based TrackingDB. A track is a feature
followed through consecutive frames; it is created when a supporter links a
previous-frame row that has no track yet, and TrackIds are issued in ascending
previous-row order (the lecturer's order, so TrackIds are unchanged).

The store keeps one sorted chunk of tracked observations per frame and is saved
as ``track_observations.npz`` (frame_ids, track_ids, u_left_px, u_right_px,
v_mean_px; sorted by frame, then track). Descriptors are not stored: no stage
after tracking reads them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..contracts import ContractError


NO_ID = -1
OBSERVATION_FIELDS = ("frame_ids", "track_ids", "u_left_px", "u_right_px", "v_mean_px")


class _GrowingArray:
    """Append-only int64 array with amortized O(1) growth."""

    def __init__(self) -> None:
        self._data = np.empty(1024, dtype=np.int64)
        self.size = 0

    def extend(self, values: np.ndarray) -> None:
        needed = self.size + len(values)
        if needed > len(self._data):
            grown = np.empty(max(needed, 2 * len(self._data)), dtype=np.int64)
            grown[: self.size] = self._data[: self.size]
            self._data = grown
        self._data[self.size:needed] = values
        self.size = needed

    def view(self) -> np.ndarray:
        return self._data[: self.size]


class TrackStore:
    """Tracks over consecutive frames, built frame by frame or loaded from arrays."""

    def __init__(self) -> None:
        self._frame_count = 0
        self._track_start = _GrowingArray()
        self._track_end = _GrowingArray()
        # Sealed frames: per-frame track ids (ascending) and their (u_l, u_r, v) links.
        self._frame_tracks = []  # type: list
        self._frame_links = []  # type: list
        # The newest frame stays open until the next frame can start tracks on it.
        self._open_row_tracks = None  # type: Optional[np.ndarray]
        self._open_links = None  # type: Optional[np.ndarray]
        self._sealed = False

    # ---------------------------------------------------------------- building
    def add_frame(
        self,
        links: np.ndarray,
        previous_rows: Optional[np.ndarray] = None,
        current_rows: Optional[np.ndarray] = None,
    ) -> int:
        """Append a frame and return its FrameId.

        ``links`` is the (rows, 3) array of (u_left, u_right, v) per stereo
        row. ``previous_rows``/``current_rows`` are the supporter pairs to the
        previous frame, ascending and unique in ``previous_rows``.
        """

        if self._sealed:
            raise ContractError("tracking store is sealed")
        links = np.asarray(links, dtype=float)
        if links.ndim != 2 or links.shape[1] != 3 or links.shape[0] == 0:
            raise ContractError("tracking store frame links must be a nonempty (N, 3) array")
        frame_id = self._frame_count
        row_tracks = np.full(links.shape[0], NO_ID, dtype=np.int64)
        if frame_id == 0:
            if previous_rows is not None or current_rows is not None:
                raise ContractError("the first tracking frame has no previous frame")
        else:
            previous_rows = np.asarray(previous_rows, dtype=np.int64)
            current_rows = np.asarray(current_rows, dtype=np.int64)
            previous_tracks = self._open_row_tracks
            if (
                previous_rows.shape != current_rows.shape
                or previous_rows.ndim != 1
                or np.any(np.diff(previous_rows) <= 0)
                or len(np.unique(current_rows)) != len(current_rows)
                or np.any(previous_rows < 0)
                or np.any(previous_rows >= len(previous_tracks))
                or np.any(current_rows < 0)
                or np.any(current_rows >= len(row_tracks))
            ):
                raise ContractError("tracking store supporter pairs are invalid")
            linked = previous_tracks[previous_rows]
            new = linked == NO_ID
            first_new = self._track_start.size
            linked[new] = np.arange(first_new, first_new + int(new.sum()), dtype=np.int64)
            previous_tracks[previous_rows] = linked
            self._track_start.extend(np.full(int(new.sum()), frame_id - 1, dtype=np.int64))
            self._track_end.extend(np.full(int(new.sum()), frame_id, dtype=np.int64))
            self._track_end.view()[linked] = frame_id
            row_tracks[current_rows] = linked
            self._seal_open_frame()
        self._open_row_tracks = row_tracks
        self._open_links = links
        self._frame_count += 1
        return frame_id

    def _seal_open_frame(self) -> None:
        tracked = np.nonzero(self._open_row_tracks != NO_ID)[0]
        order = np.argsort(self._open_row_tracks[tracked], kind="stable")
        rows = tracked[order]
        self._frame_tracks.append(self._open_row_tracks[rows])
        self._frame_links.append(self._open_links[rows])

    def seal(self) -> None:
        """Finish building: the last frame can no longer start tracks."""

        if not self._sealed and self._frame_count:
            self._seal_open_frame()
            self._open_row_tracks = None
            self._open_links = None
        self._sealed = True

    # ----------------------------------------------------------------- queries
    def frame_num(self) -> int:
        return self._frame_count

    def track_num(self) -> int:
        return self._track_start.size

    def link_num(self) -> int:
        total = sum(len(tracks) for tracks in self._frame_tracks)
        if not self._sealed and self._open_row_tracks is not None:
            total += int(np.count_nonzero(self._open_row_tracks != NO_ID))
        return total

    @property
    def last_track_id(self) -> int:
        return self.track_num() - 1

    def tracks(self, frame_id: int) -> np.ndarray:
        """Ascending TrackIds observed on ``frame_id``."""

        if not 0 <= frame_id < self._frame_count:
            return np.empty(0, dtype=np.int64)
        if frame_id < len(self._frame_tracks):
            return self._frame_tracks[frame_id]
        open_tracks = self._open_row_tracks[self._open_row_tracks != NO_ID]
        return np.sort(open_tracks, kind="stable")

    def frames(self, track_id: int) -> range:
        if not 0 <= track_id < self.track_num():
            return range(0)
        return range(
            int(self._track_start.view()[track_id]), int(self._track_end.view()[track_id]) + 1
        )

    def window_observations(
        self, first_frame_id: int, last_frame_id: int
    ) -> Dict[str, np.ndarray]:
        """Observations on sealed frames first..last, sorted by (track, frame)."""

        if not 0 <= first_frame_id <= last_frame_id < len(self._frame_tracks):
            raise ContractError("tracking store window is outside sealed frames")
        frames = range(first_frame_id, last_frame_id + 1)
        track_ids = np.concatenate([self._frame_tracks[frame] for frame in frames])
        frame_ids = np.concatenate(
            [np.full(len(self._frame_tracks[frame]), frame, dtype=np.int64) for frame in frames]
        )
        links = np.concatenate([self._frame_links[frame] for frame in frames])
        # Frames are already ascending, so a stable sort by track keeps frame order.
        order = np.argsort(track_ids, kind="stable")
        return {
            "frame_ids": frame_ids[order],
            "track_ids": track_ids[order],
            "links": links[order],
            "track_start": self._track_start.view(),
            "track_end": self._track_end.view(),
        }

    # ------------------------------------------------------------- persistence
    def observation_arrays(self) -> Dict[str, np.ndarray]:
        """All observations sorted by (frame, track)."""

        if not self._sealed:
            raise ContractError("tracking store must be sealed before saving")
        links = (
            np.concatenate(self._frame_links) if self._frame_links else np.empty((0, 3))
        )
        return {
            "frame_ids": np.concatenate(
                [np.full(len(tracks), frame, dtype=np.int64)
                 for frame, tracks in enumerate(self._frame_tracks)]
            ).astype(np.int64),
            "track_ids": np.concatenate(self._frame_tracks).astype(np.int64),
            "u_left_px": np.ascontiguousarray(links[:, 0], dtype=float),
            "u_right_px": np.ascontiguousarray(links[:, 1], dtype=float),
            "v_mean_px": np.ascontiguousarray(links[:, 2], dtype=float),
        }

    @classmethod
    def from_observation_arrays(
        cls, arrays: Mapping[str, np.ndarray], frame_count: int
    ) -> "TrackStore":
        """Rebuild a sealed store; the inverse of :meth:`observation_arrays`."""

        if tuple(sorted(arrays)) != tuple(sorted(OBSERVATION_FIELDS)):
            raise ContractError("tracking observation fields differ")
        frame_ids = np.asarray(arrays["frame_ids"], dtype=np.int64)
        track_ids = np.asarray(arrays["track_ids"], dtype=np.int64)
        links = np.stack(
            [np.asarray(arrays[name], dtype=float) for name in OBSERVATION_FIELDS[2:]], axis=1
        )
        count = len(frame_ids)
        if track_ids.shape != (count,) or links.shape != (count, 3) or frame_ids.ndim != 1:
            raise ContractError("tracking observation population differs")
        store = cls()
        store._frame_count = int(frame_count)
        bounds = np.searchsorted(frame_ids, np.arange(frame_count + 1), side="left")
        store._frame_tracks = [track_ids[a:b] for a, b in zip(bounds, bounds[1:])]
        store._frame_links = [links[a:b] for a, b in zip(bounds, bounds[1:])]
        track_count = int(track_ids.max()) + 1 if count else 0
        start = np.full(track_count, np.iinfo(np.int64).max, dtype=np.int64)
        end = np.full(track_count, -1, dtype=np.int64)
        if count:
            np.minimum.at(start, track_ids, frame_ids)
            np.maximum.at(end, track_ids, frame_ids)
        store._track_start.extend(start)
        store._track_end.extend(end)
        store._sealed = True
        validate_track_store(store, frame_count, arrays=(frame_ids, track_ids, links))
        return store

    @classmethod
    def load(cls, path: Path, frame_count: int) -> "TrackStore":
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        return cls.from_observation_arrays(arrays, frame_count)


def validate_track_store(
    store: TrackStore,
    expected_frame_count: int,
    *,
    arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    """Structural and stereo-policy invariants of a sealed store."""

    if store.frame_num() != expected_frame_count:
        raise ContractError("tracking store frame count differs")
    if arrays is None:
        observations = store.observation_arrays()
        frame_ids, track_ids = observations["frame_ids"], observations["track_ids"]
        links = np.stack([observations[name] for name in OBSERVATION_FIELDS[2:]], axis=1)
    else:
        frame_ids, track_ids, links = arrays
    if len(frame_ids) and (
        np.any(np.diff(frame_ids) < 0)
        or frame_ids[0] < 0
        or frame_ids[-1] >= expected_frame_count
    ):
        raise ContractError("tracking store FrameIds are not sorted within range")
    same_frame = np.diff(frame_ids) == 0
    if np.any(np.diff(track_ids)[same_frame] <= 0) or (len(track_ids) and track_ids.min() < 0):
        raise ContractError("tracking store TrackIds are not unique per frame")
    start = store._track_start.view()
    end = store._track_end.view()
    lengths = np.bincount(track_ids, minlength=len(start)) if len(track_ids) else np.zeros(0, int)
    if (
        len(lengths) != len(start)
        or np.any(end - start < 1)
        or not np.array_equal(lengths, end - start + 1)
    ):
        raise ContractError("tracking store contains a nonconsecutive track")
    if not np.all(np.isfinite(links)) or not np.all(links[:, 0] > links[:, 1]):
        raise ContractError("tracking observation violates stereo policy")


__all__ = ["NO_ID", "OBSERVATION_FIELDS", "TrackStore", "validate_track_store"]
