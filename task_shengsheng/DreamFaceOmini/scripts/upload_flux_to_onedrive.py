#!/usr/bin/env python3
"""Upload flux_result images to OneDrive and write public share URLs back to CSV.

Auth (pick one):
  1. ONEDRIVE_REFRESH_TOKEN + ONEDRIVE_CLIENT_ID (+ optional ONEDRIVE_CLIENT_SECRET)
  2. Interactive device-code flow (first run): omit ONEDRIVE_REFRESH_TOKEN

Env vars:
  ONEDRIVE_CLIENT_ID       Azure app client id (required)
  ONEDRIVE_CLIENT_SECRET   Optional; public client can omit
  ONEDRIVE_TENANT          Default: consumers (personal MSA). Use common/org tenant for work accounts.
  ONEDRIVE_REFRESH_TOKEN   Saved refresh token for non-interactive runs
  ONEDRIVE_FOLDER          Remote folder, default: csv-fluxout-1k
  ONEDRIVE_TOKEN_CACHE     Token cache file path

Example:
  export ONEDRIVE_CLIENT_ID=...
  python upload_flux_to_onedrive.py \\
    --csv ../exp_out/csv-fluxout-1k/csv-fluxout.csv \\
    --out ../exp_out/csv-fluxout-1k/csv-fluxout-onedrive.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import msal
import requests

GRAPH = "https://graph.microsoft.com/v1.0"
SCOPES = ["Files.ReadWrite", "offline_access"]


def load_token_cache(path: Path) -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if path.exists():
        cache.deserialize(path.read_text(encoding="utf-8"))
    return cache


def save_token_cache(path: Path, cache: msal.SerializableTokenCache) -> None:
    if cache.has_state_changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cache.serialize(), encoding="utf-8")


def acquire_token(client_id: str, client_secret: str | None, tenant: str, cache_path: Path) -> str:
    cache = load_token_cache(cache_path)
    authority = f"https://login.microsoftonline.com/{tenant}"

    if client_secret:
        app = msal.ConfidentialClientApplication(
            client_id, authority=authority, client_credential=client_secret, token_cache=cache
        )
    else:
        app = msal.PublicClientApplication(client_id, authority=authority, token_cache=cache)

    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            save_token_cache(cache_path, cache)
            return result["access_token"]

    refresh_token = os.environ.get("ONEDRIVE_REFRESH_TOKEN")
    if refresh_token:
        token_url = f"{authority}/oauth2/v2.0/token"
        data = {
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": " ".join(SCOPES),
        }
        if client_secret:
            data["client_secret"] = client_secret
        resp = requests.post(token_url, data=data, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        save_token_cache(cache_path, cache)
        return payload["access_token"]

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"Device flow failed: {flow}")
    print(flow["message"], flush=True)
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(f"Auth failed: {result}")
    save_token_cache(cache_path, cache)
    if "refresh_token" in result:
        print(f"\nSave this refresh token for later runs:\nONEDRIVE_REFRESH_TOKEN={result['refresh_token']}\n")
    return result["access_token"]


def ensure_folder(token: str, folder: str) -> None:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    parts = [p for p in folder.split("/") if p]
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        q = requests.utils.quote(f"root:/{current}:")
        resp = requests.get(f"{GRAPH}/me/drive/{q}", headers=headers, timeout=60)
        if resp.status_code == 404:
            parent_q = requests.utils.quote(f"root:/{current.rsplit('/', 1)[0]}:") if "/" in current else "root"
            create_url = f"{GRAPH}/me/drive/{parent_q}/children"
            body = {
                "name": part,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "replace",
            }
            resp = requests.post(create_url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()


def upload_file(token: str, folder: str, local_path: Path, remote_name: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    size = local_path.stat().st_size
    upload_path = requests.utils.quote(f"root:/{folder}/{remote_name}:")
    session_url = f"{GRAPH}/me/drive/{upload_path}/createUploadSession"
    session = requests.post(
        session_url,
        headers={**headers, "Content-Type": "application/json"},
        json={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
        timeout=60,
    )
    session.raise_for_status()
    upload_url = session.json()["uploadUrl"]

    chunk_size = 4 * 1024 * 1024
    with local_path.open("rb") as f:
        pos = 0
        while pos < size:
            chunk = f.read(chunk_size)
            end = pos + len(chunk) - 1
            put_headers = {
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {pos}-{end}/{size}",
            }
            resp = requests.put(upload_url, headers=put_headers, data=chunk, timeout=300)
            if resp.status_code not in (200, 201, 202):
                resp.raise_for_status()
            pos = end + 1
    return resp.json()


def create_public_link(token: str, item_id: str) -> str:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(
        f"{GRAPH}/me/drive/items/{item_id}/createLink",
        headers=headers,
        json={"type": "view", "scope": "anonymous"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["link"]["webUrl"]


def direct_download_url(token: str, item_id: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{GRAPH}/me/drive/items/{item_id}", headers=headers, timeout=60)
    resp.raise_for_status()
    item = resp.json()
    return item.get("@microsoft.graph.downloadUrl") or item["webUrl"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Input CSV with flux_result column")
    p.add_argument("--out", required=True, help="Output CSV with flux_result_url column")
    p.add_argument("--folder", default=os.environ.get("ONEDRIVE_FOLDER", "csv-fluxout-1k"))
    p.add_argument("--resume", action="store_true", help="Skip rows that already have flux_result_url in --out")
    p.add_argument("--limit", type=int, default=0, help="Upload at most N rows (debug)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    client_id = os.environ.get("ONEDRIVE_CLIENT_ID")
    if not client_id:
        print("Missing ONEDRIVE_CLIENT_ID. Register an Azure app with delegated Files.ReadWrite.", file=sys.stderr)
        return 2

    client_secret = os.environ.get("ONEDRIVE_CLIENT_SECRET")
    tenant = os.environ.get("ONEDRIVE_TENANT", "consumers")
    cache_path = Path(os.environ.get("ONEDRIVE_TOKEN_CACHE", Path.home() / ".cache/onedrive_msal.json"))

    token = acquire_token(client_id, client_secret, tenant, cache_path)
    ensure_folder(token, args.folder)

    existing: dict[str, str] = {}
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("flux_result_url"):
                    existing[row["id"]] = row["flux_result_url"]

    with Path(args.csv).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Empty CSV", file=sys.stderr)
        return 1

    fieldnames = list(rows[0].keys())
    if "flux_result_url" not in fieldnames:
        fieldnames.append("flux_result_url")
    if "onedrive_upload_status" not in fieldnames:
        fieldnames.append("onedrive_upload_status")

    uploaded = 0
    for i, row in enumerate(rows, 1):
        row_id = row.get("id", "")
        if args.resume and row_id in existing:
            row["flux_result_url"] = existing[row_id]
            row["onedrive_upload_status"] = "skipped"
            continue

        local = row.get("flux_result", "").strip()
        if not local or row.get("status") != "ok" or not Path(local).is_file():
            row["flux_result_url"] = ""
            row["onedrive_upload_status"] = "missing_file"
            continue

        remote_name = f"{row_id}.png"
        print(f"[{i}/{len(rows)}] upload {remote_name}", flush=True)
        try:
            item = upload_file(token, args.folder, Path(local), remote_name)
            url = create_public_link(token, item["id"])
            row["flux_result_url"] = url
            row["onedrive_upload_status"] = "ok"
            uploaded += 1
        except Exception as exc:  # noqa: BLE001
            row["flux_result_url"] = ""
            row["onedrive_upload_status"] = f"error: {exc}"
            print(f"  failed: {exc}", file=sys.stderr)

        if args.limit and uploaded >= args.limit:
            break

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r.get("onedrive_upload_status") == "ok")
    print(f"Done. uploaded={ok}, output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
