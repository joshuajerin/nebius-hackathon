from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from factory_sre.perception import AprilTagCameraAdapter, CameraIntrinsics, rotation_matrix_to_wxyz


def test_rotation_matrix_conversion_uses_wxyz() -> None:
    assert rotation_matrix_to_wxyz(np.eye(3)) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_intrinsics_reject_nonphysical_focal_length() -> None:
    with pytest.raises(ValueError, match="focal"):
        CameraIntrinsics(0.0, 500.0, 320.0, 240.0)


def test_generated_coarse_tag_is_detectable_with_metric_pose() -> None:
    tag = np.asarray(Image.open("assets/tags/tag36h11_10.png").convert("L").resize((256, 256)))
    frame = np.full((512, 512), 255, dtype=np.uint8)
    frame[128:384, 128:384] = tag
    adapter = AprilTagCameraAdapter(
        lambda: frame,
        CameraIntrinsics(500.0, 500.0, 256.0, 256.0),
        decision_margin=5.0,
    )
    detections = adapter.detect_tags("tag36h11", 0.120)
    assert len(detections) == 1
    assert detections[0][0] == 10
    assert detections[0][1].translation[2] > 0
