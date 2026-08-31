#!/usr/bin/env python3
"""Upload flux_result PNGs to OneDrive via rclone and write public share URLs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

RCLONE = "/tmp/rclone-v1.74.3-linux-amd64/rclone"
REMOTE_DIR = "onedrive:csv-fluxout-1k"
CSV_IN = Path("/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv")
CSV_OUT = Path("/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout-onedrive.csv")
STATE = CSV_OUT.with_suffix(".state.json")


def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def load_state() -> dict[str, str]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, str]) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def remote_exists(remote_path: str) -> bool:
    proc = run([RCLONE, "lsf", remote_path])
    return proc.returncode == 0 and proc.stdout.strip() != ""


def upload_file(local_path: Path, remote_name: str) -> bool:
    remote = f"{REMOTE_DIR}/{remote_name}"
    if remote_exists(remote):
        return True
    proc = run([RCLONE, "copyto", str(local_path), remote, "--retries", "3"], timeout=900)
    if proc.returncode != 0:
        print(f"upload failed {remote_name}: {proc.stderr.strip()}", file=sys.stderr)
        return False
    return True


def create_link(remote_name: str) -> str:
    remote = f"{REMOTE_DIR}/{remote_name}"
    proc = run([RCLONE, "link", remote], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def main() -> int:
    run([RCLONE, "mkdir", REMOTE_DIR])

    with CSV_IN.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty input csv", file=sys.stderr)
        return 1

    state = load_state()
    fieldnames = list(rows[0].keys())
    for col in ("flux_result_url", "onedrive_upload_status"):
        if col not in fieldnames:
            fieldnames.append(col)

    total = len(rows)
    for i, row in enumerate(rows, 1):
        row_id = row.get("id", "")
        if row_id in state:
            row["flux_result_url"] = state[row_id]
            row["onedrive_upload_status"] = "ok"
            continue

        local = row.get("flux_result", "").strip()
        if row.get("status") != "ok" or not local or not Path(local).is_file():
            row["flux_result_url"] = ""
            row["onedrive_upload_status"] = "skipped"
            continue

        remote_name = f"{row_id}.png"
        print(f"[{i}/{total}] {remote_name}", flush=True)
        if not upload_file(Path(local), remote_name):
            row["flux_result_url"] = ""
            row["onedrive_upload_status"] = "upload_failed"
            continue

        try:
            url = create_link(remote_name)
            state[row_id] = url
            save_state(state)
            row["flux_result_url"] = url
            row["onedrive_upload_status"] = "ok"
            print(f"  ok: {url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["flux_result_url"] = ""
            row["onedrive_upload_status"] = f"link_failed: {exc}"
            print(f"  link failed: {exc}", file=sys.stderr)

        if i % 5 == 0:
            with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r.get("onedrive_upload_status") == "ok")
    print(f"done: {ok}/{total} -> {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
