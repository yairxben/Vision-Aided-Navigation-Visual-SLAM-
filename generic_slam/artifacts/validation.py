"""Common structural validation helpers."""

from __future__ import annotations

import numpy as np

from ..contracts import ContractError


def require_finite_array(value: np.ndarray, shape: tuple, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ContractError("{} has invalid shape or nonfinite values".format(name))
    return array


def require_strictly_increasing(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or not len(array) or np.any(np.diff(array) <= 0):
        raise ContractError("{} must be a nonempty strictly increasing vector".format(name))
    return array


__all__ = ["require_finite_array", "require_strictly_increasing"]

