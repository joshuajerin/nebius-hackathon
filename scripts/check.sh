#!/usr/bin/env bash
set -euo pipefail

uv run --extra dev pytest -q
uv run python -m compileall -q factory_sre tests
uv run antioch scenario collect --json
