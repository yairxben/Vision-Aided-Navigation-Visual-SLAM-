"""Sequence-generic KITTI stereo SLAM pipeline.

The package is fully independent of the frozen course code in
``course_exercises/``; feature tracks live in the columnar
:class:`generic_slam.tracking.store.TrackStore`.
"""

from .config import AlgorithmConfig, resolve_algorithm_config
from .contracts import RunRequest

__all__ = ["AlgorithmConfig", "RunRequest", "resolve_algorithm_config"]

