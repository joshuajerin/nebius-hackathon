from __future__ import annotations

from dataclasses import replace

import pytest

from factory_sre.sim.circular_effector import ASSET_PATH, DEFAULT_CIRCULAR_EFFECTOR


def test_magnetic_circle_effector_asset_contract() -> None:
    DEFAULT_CIRCULAR_EFFECTOR.validate()
    source = ASSET_PATH.read_text()

    assert 'defaultPrim = "CircularServiceEffector"' in source
    assert 'factorySre:assetRole = "circular_service_effector"' in source
    assert 'def Cylinder "ContactFace"' in source
    assert 'factorySre:frameRole = "magnetic_contact"' in source
    assert 'factorySre:contactNormal = (1, 0, 0)' in source
    assert 'uniform token info:id = "UsdPreviewSurface"' in source
    assert source.count('prepend apiSchemas = ["PhysicsCollisionAPI"]') == 3


def test_magnetic_circle_effector_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="contact face"):
        replace(DEFAULT_CIRCULAR_EFFECTOR, contact_radius_m=0.060).validate()

    with pytest.raises(ValueError, match="positive"):
        replace(DEFAULT_CIRCULAR_EFFECTOR, radius_m=0.0).validate()
