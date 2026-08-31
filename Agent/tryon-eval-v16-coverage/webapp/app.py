"""FastAPI app for the standalone human-feedback review.

``build_app(run_dir)`` binds the app to one evaluation run directory
(``runs/<ts>/``) containing ``detection.json`` and
``human/review_model.json``. The React front-end (byte-copied from the ERA
build, same ``/api/*`` contract) is served at ``/``.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from evaluator.review_model import load_review_model, write_json_atomic
from .data import derive_human_labels_from_model
from .images import build_sample_index, load_image, prewarm
from .store import (
    feedback_path,
    human_labels_path,
    read_feedback,
    set_comparison_mark,
    set_general,
    set_item_mark,
    write_feedback,
)

STATIC_DIR = Path(__file__).parent / "static"

_PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8">
<title>Try-On Eval — Human Feedback</title></head><body style="font-family:sans-serif;
max-width:40rem;margin:4rem auto;line-height:1.6">
<h1>Try-On Eval — Human Feedback</h1>
<p>The review front-end bundle is missing (webapp/static/). The API is live —
see <a href="/api/review">/api/review</a>.</p>
</body></html>"""


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_app(run_dir: str | Path) -> FastAPI:
    run_dir = Path(run_dir).resolve()
    detection = json.loads((run_dir / "detection.json").read_text(encoding="utf-8"))
    rm = load_review_model(run_dir)
    if rm is None:
        raise SystemExit(f"{run_dir}/human/review_model.json is missing — "
                         f"run the evaluation first (or --review-only on a "
                         f"finished run)")
    iteration = int(rm.get("iteration") or 1)
    mode = rm.get("mode") or "full"
    sample_index = build_sample_index(detection)
    sample_keys = [s.get("sample_key") for s in rm.get("samples", [])]
    lock = threading.Lock()

    # Warm the image cache in the background so the first review is fast — the
    # dataset images sit on a slow network mount.
    threading.Thread(
        target=prewarm, args=(sample_index, detection, sample_keys), daemon=True,
    ).start()

    app = FastAPI(title="Try-On Eval Human Feedback", docs_url="/api/docs")

    # ---- API -----------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "iteration": iteration,
                "iteration_dir": run_dir.name, "mode": mode}

    @app.get("/api/review")
    def review() -> dict:
        fb = read_feedback(run_dir, iteration, mode)
        return {**rm, "feedback": fb}

    @app.get("/api/image")
    def image(method: str, sample: str, role: str):
        loaded = load_image(sample_index, method, sample, role)
        if loaded is None:
            raise HTTPException(status_code=404, detail="image not found")
        data, media_type = loaded
        return Response(
            content=data, media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/overview")
    def overview() -> dict:
        summary_path = run_dir / "results" / "summary.json"
        summary = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = None
        return {"summary": summary, "comparison": None,
                "final_comparison": None, "diagnostics": None}

    @app.put("/api/feedback/item")
    def put_item(body: dict = Body(...)) -> dict:
        for key in ("sample_key", "method_id", "combination_id", "label"):
            if not body.get(key):
                raise HTTPException(status_code=422, detail=f"missing {key}")
        with lock:
            fb = read_feedback(run_dir, iteration, mode)
            set_item_mark(fb, body["sample_key"], body["method_id"],
                          body["combination_id"], body["label"],
                          body.get("comment", ""))
            write_feedback(run_dir, fb)
        return {"status": "ok", "feedback": fb}

    @app.put("/api/feedback/comparison")
    def put_comparison(body: dict = Body(...)) -> dict:
        for key in ("sample_key", "combination_id", "label"):
            if not body.get(key):
                raise HTTPException(status_code=422, detail=f"missing {key}")
        with lock:
            fb = read_feedback(run_dir, iteration, mode)
            set_comparison_mark(fb, body["sample_key"], body["combination_id"],
                                body["label"], body.get("comment", ""))
            write_feedback(run_dir, fb)
        return {"status": "ok", "feedback": fb}

    @app.put("/api/feedback/general")
    def put_general(body: dict = Body(...)) -> dict:
        with lock:
            fb = read_feedback(run_dir, iteration, mode)
            set_general(fb, body.get("text", ""))
            write_feedback(run_dir, fb)
        return {"status": "ok", "feedback": fb}

    @app.post("/api/feedback/save")
    def save() -> dict:
        with lock:
            fb = read_feedback(run_dir, iteration, mode)
            write_feedback(run_dir, fb)
        return {"status": "saved", "updated_at": fb.get("updated_at")}

    @app.post("/api/feedback/finalize")
    def finalize(body: dict = Body(default={})) -> dict:
        reopen = bool((body or {}).get("reopen"))
        with lock:
            fb = read_feedback(run_dir, iteration, mode)
            if fb.get("status") == "finalized" and not reopen:
                raise HTTPException(
                    status_code=409,
                    detail="feedback already finalized — pass reopen:true to re-finalize",
                )
            prior_final = (fb.get("finalized_at")
                           if fb.get("status") == "finalized" else None)
            fb["status"] = "finalized"
            fb["finalized_at"] = prior_final or fb.get("finalized_at") or _now_iso()
            write_feedback(run_dir, fb)
            labels = derive_human_labels_from_model(rm, fb)
            write_json_atomic(human_labels_path(run_dir), labels)
        return {
            "status": "ok",
            "iteration_dir": run_dir.name,
            "feedback_json": str(feedback_path(run_dir)),
            "human_labels_json": str(human_labels_path(run_dir)),
            "labels": {
                "flagging_label_count": len(labels["labels"]),
                "comparison_label_count": len(labels["comparison_labels"]),
                "config_summary": labels["config_summary"],
            },
        }

    # ---- static front-end ---------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_html = STATIC_DIR / "index.html"
        if index_html.is_file():
            return FileResponse(index_html)
        return HTMLResponse(_PLACEHOLDER)

    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
    return app
