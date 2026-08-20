"""Configuration hashing and telemetry completeness contracts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TelemetrySchedule:
    joint_force_hz: int = 20
    state_pose_health_hz: int = 10
    inspection_rgb_hz: int = 5

    def __post_init__(self) -> None:
        if (self.joint_force_hz, self.state_pose_health_hz, self.inspection_rgb_hz) != (20, 10, 5):
            raise ValueError("certification telemetry rates must remain 20/10/5 Hz")


@dataclass(frozen=True, slots=True)
class EvidenceManifest:
    antioch_version: str
    isaac_sim_version: str
    go2_asset: str
    go2_policy: str
    so101_asset: str
    project_file_sha256: str
    calibration_sha256: str | None
    schedule: TelemetrySchedule = TelemetrySchedule()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_evidence_manifest(project_root: str | Path, calibration_path: str | Path | None = None) -> EvidenceManifest:
    root = Path(project_root)
    calibration_hash = None
    if calibration_path is not None and Path(calibration_path).is_file():
        calibration_hash = sha256_file(calibration_path)
    return EvidenceManifest(
        antioch_version="0.3.63",
        isaac_sim_version="6.0.1",
        go2_asset="/Isaac/Samples/Policies/go2/go2.usda",
        go2_policy="/Isaac/Samples/Policies/go2/physx_policy.pt",
        so101_asset="so101_antioch@1.3.2",
        project_file_sha256=sha256_file(root / "antioch.yaml"),
        calibration_sha256=calibration_hash,
    )
