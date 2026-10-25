"""Runtime KITTI P0/P1 calibration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from ..contracts import ContractError
from ..geometry import homogeneous, validate_rigid_pose


@dataclass(frozen=True)
class StereoCalibration:
    p0: np.ndarray
    p1: np.ndarray
    intrinsic: np.ndarray
    left_extrinsic: np.ndarray
    right_extrinsic: np.ndarray
    right_from_left: np.ndarray
    fx_px: float
    fy_px: float
    skew_px: float
    cx_px: float
    cy_px: float
    baseline_m: float

    def record(self) -> Dict[str, object]:
        return {
            "p0": self.p0.tolist(),
            "p1": self.p1.tolist(),
            "intrinsic": self.intrinsic.tolist(),
            "left_extrinsic": self.left_extrinsic.tolist(),
            "right_extrinsic": self.right_extrinsic.tolist(),
            "right_from_left": self.right_from_left.tolist(),
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "skew_px": self.skew_px,
            "cx_px": self.cx_px,
            "cy_px": self.cy_px,
            "baseline_m": self.baseline_m,
            "stereo_measurement_order": ["u_left_px", "u_right_px", "v_mean_px"],
        }


def _projection_rows(path: Path) -> Mapping[str, np.ndarray]:
    rows: Dict[str, np.ndarray] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError("cannot read calibration: {}".format(error))
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        label = tokens[0].rstrip(":")
        if label not in ("P0", "P1"):
            continue
        if len(tokens) != 13:
            raise ContractError("{} must contain twelve calibration values".format(label))
        try:
            matrix = np.asarray(tokens[1:], dtype=float).reshape(3, 4)
        except ValueError as error:
            raise ContractError("invalid {} calibration row: {}".format(label, error))
        if not np.all(np.isfinite(matrix)):
            raise ContractError("{} calibration contains nonfinite values".format(label))
        rows[label] = matrix
    if set(rows) != {"P0", "P1"}:
        raise ContractError("calibration must contain P0 and P1")
    return rows


def load_stereo_calibration(path: Path, atol: float = 1e-6) -> StereoCalibration:
    rows = _projection_rows(Path(path))
    p0 = rows["P0"]
    p1 = rows["P1"]
    intrinsic = p0[:, :3]
    if abs(np.linalg.det(intrinsic)) <= np.finfo(float).eps:
        raise ContractError("P0 intrinsic matrix is singular")
    if (
        intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], rtol=0.0, atol=atol)
    ):
        raise ContractError("P0 must contain a valid positive pinhole intrinsic")
    inverse_intrinsic = np.linalg.inv(intrinsic)
    left = inverse_intrinsic @ p0
    right = inverse_intrinsic @ p1
    left_h = validate_rigid_pose(homogeneous(left), atol=atol)
    right_h = validate_rigid_pose(homogeneous(right), atol=atol)
    # Preserve the validated course convention exactly: P0 defines K and the
    # normalized P1 extrinsic is the fixed right-camera transform used by the
    # stereo kernels. KITTI odometry P0 is the left reference camera (identity
    # up to floating-point inversion noise).
    if not np.allclose(left_h, np.eye(4), rtol=0.0, atol=atol):
        raise ContractError("P0 must define the rectified left reference camera")
    right_from_left = right_h.copy()
    if not np.allclose(right_from_left[:3, :3], np.eye(3), rtol=0.0, atol=atol):
        raise ContractError("P0/P1 cameras are not rectified")
    if not np.allclose(right_from_left[1:3, 3], 0.0, rtol=0.0, atol=atol):
        raise ContractError("P0/P1 baseline is not horizontal")
    baseline = -float(right_from_left[0, 3])
    if not math_is_positive_finite(baseline):
        raise ContractError("P0/P1 must produce a positive stereo baseline")
    return StereoCalibration(
        p0=p0.copy(),
        p1=p1.copy(),
        intrinsic=intrinsic.copy(),
        left_extrinsic=left_h,
        right_extrinsic=right_h,
        right_from_left=right_from_left,
        fx_px=float(intrinsic[0, 0]),
        fy_px=float(intrinsic[1, 1]),
        skew_px=float(intrinsic[0, 1]),
        cx_px=float(intrinsic[0, 2]),
        cy_px=float(intrinsic[1, 2]),
        baseline_m=baseline,
    )


def math_is_positive_finite(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


__all__ = ["StereoCalibration", "load_stereo_calibration"]
