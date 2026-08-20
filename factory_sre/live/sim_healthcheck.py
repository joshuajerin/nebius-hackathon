"""Docker healthcheck for the persistent simulator's live-frame heartbeat."""

from __future__ import annotations

import os
import time
from pathlib import Path


def main() -> None:
    frame = Path(
        os.environ.get("FACTORY_SRE_FRAME_PATH", "/workspace/output/live/latest.jpg")
    )
    if not frame.is_file() or frame.stat().st_size < 1024:
        raise SystemExit(1)
    if time.time() - frame.stat().st_mtime > 30.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
