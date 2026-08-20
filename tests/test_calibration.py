from __future__ import annotations

import json

import pytest

from factory_sre.calibration import load_calibration


def test_example_calibration_fails_closed() -> None:
    with pytest.raises(ValueError, match="not approved"):
        load_calibration("config/calibration.example.json")


def test_approved_calibration_loads(tmp_path) -> None:
    payload = json.loads(open("config/calibration.example.json").read())
    payload["calibrated"] = True
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    calibration = load_calibration(path)
    assert calibration.fine_tag_to_usb_port.translation == (-0.06, 0.0, 0.0)
