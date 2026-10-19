"""Sequence-generic KITTI stereo SLAM pipeline.

The package is intentionally independent of the historical exercise-oriented
entrypoints.  The lecturer TrackingDB is the sole shared implementation
component and is wrapped under :mod:`generic_slam.tracking`.
"""

from .config import AlgorithmConfig, resolve_algorithm_config
from .contracts import RunRequest

__all__ = ["AlgorithmConfig", "RunRequest", "resolve_algorithm_config"]

