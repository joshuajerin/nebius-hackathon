"""AprilTag pose estimation for the certified SICK InspectorP61x wrist camera."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pupil_apriltags import Detector

from .contracts import Transform3D


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")


def rotation_matrix_to_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""

    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation matrix, got {m.shape}")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("rotation matrix produced a zero quaternion")
    return tuple(float(value / norm) for value in q)


class AprilTagCameraAdapter:
    """Capture frames on demand and estimate metric tag poses in the camera frame."""

    def __init__(
        self,
        capture: Callable[[], object],
        intrinsics: CameraIntrinsics,
        *,
        family: str = "tag36h11",
        decision_margin: float = 25.0,
    ) -> None:
        self._capture = capture
        self.intrinsics = intrinsics
        self.family = family
        self.decision_margin = decision_margin
        self._detector = Detector(families=family, nthreads=2, quad_decimate=1.0, refine_edges=1)

    def capture_rgb(self) -> object:
        return self._capture()

    def detect_tags(self, family: str, size_m: float) -> tuple[tuple[int, Transform3D], ...]:
        if family != self.family or size_m <= 0:
            return ()
        frame = np.asarray(self._capture())
        if frame.ndim == 3:
            frame = np.asarray(Image.fromarray(frame[..., :3].astype(np.uint8)).convert("L"))
        elif frame.ndim != 2:
            raise ValueError(f"camera frame must be HxW or HxWxC, got {frame.shape}")
        detections = self._detector.detect(
            np.ascontiguousarray(frame.astype(np.uint8)),
            estimate_tag_pose=True,
            camera_params=(self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy),
            tag_size=size_m,
        )
        result: list[tuple[int, Transform3D]] = []
        for detection in detections:
            if float(detection.decision_margin) < self.decision_margin:
                continue
            translation = tuple(float(value) for value in np.asarray(detection.pose_t).reshape(3))
            quaternion = rotation_matrix_to_wxyz(np.asarray(detection.pose_R))
            result.append((int(detection.tag_id), Transform3D(translation, quaternion)))
        return tuple(result)
