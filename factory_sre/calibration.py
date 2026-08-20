"""Fail-closed loading of physical camera, mount, tool, and port calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import Transform3D
from .perception import CameraIntrinsics


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    wrist_camera: CameraIntrinsics
    body_to_arm_mount: Transform3D
    wrist_camera_to_usb_tip: Transform3D
    fine_tag_to_usb_port: Transform3D


def _transform(payload: dict[str, object]) -> Transform3D:
    translation = tuple(float(value) for value in payload["translation_m"])
    quaternion = tuple(float(value) for value in payload["quaternion_wxyz"])
    if len(translation) != 3 or len(quaternion) != 4:
        raise ValueError("calibration transforms require 3-vector translation and 4-vector WXYZ quaternion")
    return Transform3D(translation, quaternion)


def load_calibration(path: str | Path) -> CalibrationBundle:
    payload = json.loads(Path(path).read_text())
    if payload.get("calibrated") is not True:
        raise ValueError("calibration is not approved; set calibrated=true only after physical measurement")
    intrinsics = payload["wrist_camera_intrinsics"]
    return CalibrationBundle(
        wrist_camera=CameraIntrinsics(
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
        ),
        body_to_arm_mount=_transform(payload["body_to_arm_mount"]),
        wrist_camera_to_usb_tip=_transform(payload["wrist_camera_to_usb_tip"]),
        fine_tag_to_usb_port=_transform(payload["fine_tag_to_usb_port"]),
    )
