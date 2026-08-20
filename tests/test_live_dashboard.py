from __future__ import annotations

from pathlib import Path


def test_dashboard_contains_real_fleet_controls_and_live_frame() -> None:
    source = Path("factory_sre/live/dashboard/index.html").read_text()

    assert "Factory SRE Control" in source
    assert "/v1/fleet" in source
    assert "/v1/sim/frame.jpg" in source
    assert "Inject 3 failures" in source
    assert "service-box-2" in source
    assert "service-box-5" in source
    assert "service-box-7" in source
    assert "5.0 s" in source
