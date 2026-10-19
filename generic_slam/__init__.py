"""Sequence-generic KITTI stereo SLAM pipeline.

The package is fully independent of the frozen course code in
``course_exercises/``.  The lecturer TrackingDB is vendored and wrapped under
:mod:`generic_slam.tracking`.
"""

from .config import AlgorithmConfig, resolve_algorithm_config
from .contracts import RunRequest

__all__ = ["AlgorithmConfig", "RunRequest", "resolve_algorithm_config"]

