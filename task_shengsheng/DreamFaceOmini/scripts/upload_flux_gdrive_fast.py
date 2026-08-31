#!/usr/bin/env python3
"""Fast parallel Google Drive upload + Sheets-friendly direct image URLs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RCLONE = "/tmp/rclone-v1.74.3-linux-amd64/rclone"
REMOTE = "gdrive"
REMOTE_DIR = f"{REMOTE}:csv-fluxout-1k"
STAGING = Path("/tmp/flux_gdrive_staging")
CSV_IN = Path("/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv")
CSV_OUT = Path("/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout-gdrive.csv")
STATE = CSV_OUT.with_suffix(".state.json")
TRANSFERS = 8
LINK_WORKERS = 4


def sheets_image_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=view&id={file_id}"


def run(cmd: list[str], timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def load_state() -> dict[str, str]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, str]) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_file_id(remote_name: str) -> str:
    remote = f"{REMOTE_DIR}/{remote_name}"
    proc = run([RCLONE, "lsjson", remote], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    items = json.loads(proc.stdout or "[]")
    if not items:
        raise RuntimeError("file not found on drive")
    return items[0]["ID"]


def make_public_and_url(row_id: str) -> tuple[str, str | None, str | None]:
    remote_name = f"{row_id}.png"
    remote = f"{REMOTE_DIR}/{remote_name}"
    proc = run([RCLONE, "link", remote], timeout=120)
    if proc.returncode != 0:
        return row_id, None, (proc.stderr or proc.stdout).strip()
    try:
        file_id = get_file_id(remote_name)
        return row_id, sheets_image_url(file_id), None
    except Exception as exc:  # noqa: BLE001
        return row_id, None, str(exc)


def main() -> int:
    test = run([RCLONE, "about", f"{REMOTE}:"], timeout=60)
    if test.returncode != 0:
        print(
            "Google Drive remote 'gdrive' is not configured.\n"
            "On your Mac run: rclone authorize \"drive\"\n"
            "Then save token with scripts/setup_gdrive_rclone.py",
            file=sys.stderr,
        )
        print(test.stderr, file=sys.stderr)
        return 2

    with CSV_IN.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    jobs: list[tuple[str, Path]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        local = Path(row.get("flux_result", "").strip())
        if local.is_file():
            jobs.append((row["id"], local))

    print(f"prepare staging for {len(jobs)} files", flush=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    for p in STAGING.iterdir():
        p.unlink()
    for row_id, local in jobs:
        target = STAGING / f"{row_id}.png"
        if not target.exists():
            target.symlink_to(local)

    run([RCLONE, "mkdir", REMOTE_DIR])
    print(f"bulk upload with transfers={TRANSFERS}", flush=True)
    proc = run(
        [
            RCLONE,
            "copy",
            str(STAGING),
            REMOTE_DIR,
            "--copy-links",
            "--transfers",
            str(TRANSFERS),
            "--checkers",
            str(TRANSFERS),
            "--drive-chunk-size",
            "32M",
            "--retries",
            "3",
            "--stats",
            "10s",
            "--stats-one-line",
        ],
        timeout=7200,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(proc.stdout)
        return 1

    remote_count = len([x for x in run([RCLONE, "lsf", REMOTE_DIR]).stdout.splitlines() if x.strip()])
    print(f"remote files: {remote_count}/{len(jobs)}", flush=True)

    state = load_state()
    pending = [row_id for row_id, _ in jobs if row_id not in state]
    print(f"make public + build urls: {len(pending)} pending", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=LINK_WORKERS) as pool:
        futures = {pool.submit(make_public_and_url, row_id): row_id for row_id in pending}
        for fut in as_completed(futures):
            row_id, url, err = fut.result()
            done += 1
            if url:
                state[row_id] = url
                if done % 20 == 0:
                    save_state(state)
                print(f"[{done}/{len(pending)}] ok {row_id}", flush=True)
            else:
                print(f"[{done}/{len(pending)}] fail {row_id}: {err}", file=sys.stderr)

    save_state(state)

    fieldnames = list(rows[0].keys())
    for col in ("flux_result_gdrive_url", "gdrive_upload_status"):
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        row_id = row.get("id", "")
        url = state.get(row_id, "")
        row["flux_result_gdrive_url"] = url
        if row.get("status") == "ok" and url:
            row["gdrive_upload_status"] = "ok"
        elif row.get("status") == "ok":
            row["gdrive_upload_status"] = "failed"
        else:
            row["gdrive_upload_status"] = "skipped"

    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r.get("gdrive_upload_status") == "ok")
    print(f"done: {ok}/{len(jobs)} -> {CSV_OUT}")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
