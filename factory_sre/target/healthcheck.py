"""Container health probe for the always-on control listener."""

from __future__ import annotations

import os
import urllib.request


def main() -> None:
    port = int(os.environ.get("FACTORY_SRE_CONTROL_PORT", "8765"))
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as response:
        if response.status != 200:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
