#!/usr/bin/env python3
"""Serve the comparison UI and persist human evaluation results."""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse


WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
DATA_DIR = WEB_DIR / "data"
RESULTS_PATH = DATA_DIR / "human_evaluations.json"


def make_pair_key(models: list[str]) -> str:
    return "::".join(sorted(models))


def make_eval_key(sample_key: str, models: list[str]) -> str:
    return f"{sample_key}::{make_pair_key(models)}"


def load_results() -> dict[str, Any]:
    if not RESULTS_PATH.exists():
        return {"evaluations": {}}
    try:
        with RESULTS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {"evaluations": {}}
    if not isinstance(data, dict) or not isinstance(data.get("evaluations"), dict):
        return {"evaluations": {}}
    return data


def write_results(data: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with NamedTemporaryFile("w", encoding="utf-8", dir=DATA_DIR, delete=False) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(RESULTS_PATH)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/human-evaluations":
            self.write_json(load_results())
            return
        if parsed.path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/web/index.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/human-evaluations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            evaluation = self.normalize_evaluation(payload)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        data = load_results()
        data.setdefault("evaluations", {})[evaluation["id"]] = evaluation
        write_results(data)
        self.write_json({"ok": True, "evaluation": evaluation})

    def normalize_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_key = str(payload.get("sample_key", "")).strip()
        models = payload.get("models")
        winner = str(payload.get("winner", "")).strip()

        if not sample_key:
            raise ValueError("sample_key is required")
        if not isinstance(models, list) or len(models) != 2:
            raise ValueError("models must contain exactly two model names")
        normalized_models = [str(model).strip() for model in models]
        if not all(normalized_models) or normalized_models[0] == normalized_models[1]:
            raise ValueError("models must be two different non-empty names")
        if winner != "tie" and winner not in normalized_models:
            raise ValueError("winner must be one of the selected models or tie")

        now = datetime.now().isoformat(timespec="seconds")
        return {
            "id": make_eval_key(sample_key, normalized_models),
            "sample_key": sample_key,
            "pair_key": make_pair_key(normalized_models),
            "models": normalized_models,
            "winner": winner,
            "updated_at": now,
        }

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GEditBench comparison web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    mimetypes.add_type("application/javascript", ".js")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {ROOT} at http://{args.host}:{args.port}/web/index.html")
    print(f"Human evaluations will be saved to {RESULTS_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
