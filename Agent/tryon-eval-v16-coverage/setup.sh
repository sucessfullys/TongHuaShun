#!/usr/bin/env bash
# Create this method's OWN virtualenv (<pkg>/.venv) and install its deps.
# Idempotent: re-running is a fast no-op once the .bootstrapped sentinel matches.
# Pass --light to skip the optional metric deps (torch/torchvision/timm).
set -e
cd "$(dirname "$0")"
exec python3 run.py --bootstrap-only "$@"
