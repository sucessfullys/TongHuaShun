#!/usr/bin/env python3
"""Fast parallel OneDrive upload via rclone + parallel link generation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RCLONE = "/tmp/rclone-v1.74.3-linux-amd64/rclone"
REMOTE_DIR = "onedrive:csv-fluxout-1k"
STAGING = Path("/tmp/flux_onedrive_staging")
CSV_IN = Path("/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv")
CSV_OUT = Path("/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout-onedrive.csv")
STATE = CSV_OUT.with_suffix(".state.json")
TRANSFERS = 16
LINK_WORKERS = 6


def run(cmd: list[str], timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def load_state() -> dict[str, str]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, str]) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def make_link(row_id: str) -> tuple[str, str | None, str | None]:
    remote = f"{REMOTE_DIR}/{row_id}.png"
    proc = run([RCLONE, "link", remote], timeout=120)
    if proc.returncode != 0:
        return row_id, None, (proc.stderr or proc.stdout).strip()
    return row_id, proc.stdout.strip(), None


def main() -> int:
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
    if STAGING.exists():
        for p in STAGING.iterdir():
            p.unlink()
    else:
        STAGING.mkdir(parents=True)
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
            "--retries",
            "3",
            "--low-level-retries",
            "10",
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

    remote_count = run([RCLONE, "lsf", REMOTE_DIR])
    uploaded = len([x for x in remote_count.stdout.splitlines() if x.strip()])
    print(f"remote files: {uploaded}/{len(jobs)}", flush=True)
    if uploaded < len(jobs):
        print("upload incomplete, retrying missing files once", flush=True)
        proc = run(
            [
                RCLONE,
                "copy",
                str(STAGING),
                REMOTE_DIR,
                "--copy-links",
                "--ignore-existing",
                "--transfers",
                str(TRANSFERS),
            ],
            timeout=7200,
        )
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return 1

    state = load_state()
    pending = [row_id for row_id, _ in jobs if row_id not in state]
    print(f"generate links: {len(pending)} pending, workers={LINK_WORKERS}", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=LINK_WORKERS) as pool:
        futures = {pool.submit(make_link, row_id): row_id for row_id in pending}
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
    for col in ("flux_result_url", "onedrive_upload_status"):
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        row_id = row.get("id", "")
        url = state.get(row_id, "")
        row["flux_result_url"] = url
        if row.get("status") == "ok" and url:
            row["onedrive_upload_status"] = "ok"
        elif row.get("status") == "ok":
            row["onedrive_upload_status"] = "link_failed"
        else:
            row["onedrive_upload_status"] = "skipped"

    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r.get("onedrive_upload_status") == "ok")
    print(f"done: {ok}/{len(jobs)} -> {CSV_OUT}")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
