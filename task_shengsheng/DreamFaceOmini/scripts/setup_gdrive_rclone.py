#!/usr/bin/env python3
"""Configure rclone 'gdrive' remote from Mac authorize JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONF = Path("/root/.config/rclone/rclone.conf")
TOKEN_PATH = Path("/tmp/gdrive_token.json")


def main() -> int:
    if len(sys.argv) > 1:
        token = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    elif TOKEN_PATH.exists():
        token = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    else:
        print("Usage: setup_gdrive_rclone.py [/path/to/token.json]", file=sys.stderr)
        print("Or paste authorize JSON into /tmp/gdrive_token.json", file=sys.stderr)
        return 2

    CONF.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if CONF.exists():
        keep = []
        in_gdrive = False
        for line in CONF.read_text(encoding="utf-8").splitlines():
            if line.startswith("[gdrive]"):
                in_gdrive = True
                continue
            if in_gdrive and line.startswith("["):
                in_gdrive = False
            if not in_gdrive:
                keep.append(line)
        lines = keep

    block = [
        "[gdrive]",
        "type = drive",
        "scope = drive",
        f"token = {json.dumps(token, separators=(',', ':'))}",
        "",
    ]
    CONF.write_text("\n".join(lines + block), encoding="utf-8")
    print(f"configured gdrive in {CONF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
