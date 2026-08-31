"""FastAPI app for the ERA Stage 8 human-feedback review web app.

``build_app(workspace_path, iteration)`` returns a FastAPI app bound to one
iteration's experiment results. The React front-end (built into ``static/``) is
served at ``/``; the review data and feedback API live under ``/api``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config import ERAConfig
from ..orchestration.human_feedback import finalize_feedback
from ..orchestration.review_adapter import load_review_model
from ..workspace import Workspace, resolve_workspace_root
from .data import build_review_model, review_payload
from .images import load_image, prewarm
from .store import (
    read_feedback,
    set_comparison_mark,
    set_general,
    set_item_mark,
    write_feedback,
)

STATIC_DIR = Path(__file__).parent / "static"

_PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8">
<title>ERA — Human Feedback</title></head><body style="font-family:sans-serif;
max-width:40rem;margin:4rem auto;line-height:1.6">
<h1>ERA — Human Feedback</h1>
<p>The review front-end has not been built yet. From the ERA repo root run:</p>
<pre>cd era/webapp/frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>The API is live — see <a href="/api/review">/api/review</a>.</p>
</body></html>"""


def _resolve_iteration(workspace_path: str, iteration: object) -> tuple[Path, Path]:
    """Return ``(workspace_root, iter_dir)`` for a workspace + optional iteration."""
    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if iteration in (None, ""):
        return root, ws.iter_path()
    text = str(iteration).strip()
    if text.startswith("iter_"):
        text = text[len("iter_"):]
    return root, root / f"iter_{int(text):03d}"


def build_app(workspace_path: str, iteration: object = None) -> FastAPI:
    """Build the FastAPI app for one ERA iteration's human-feedback review."""
    workspace_root, iter_dir = _resolve_iteration(workspace_path, iteration)
    cfg = ERAConfig.from_yaml(workspace_root / "config.yaml")
    model = build_review_model(workspace_root, iter_dir, cfg)
    lock = threading.Lock()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Warm the image cache in the background so the first review is fast — the
    # dataset images sit on a slow network mount.
    threading.Thread(
        target=prewarm, args=(cfg, model.sample_keys), daemon=True,
    ).start()

    app = FastAPI(title="ERA Human Feedback", docs_url="/api/docs")

    # ---- API -----------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "iteration": model.iteration,
                "iteration_dir": iter_dir.name, "mode": model.mode}

    @app.get("/api/review")
    def review() -> dict:
        fb = read_feedback(iter_dir, model.iteration, model.mode)
        # Prefer the Stage 8 adapter's normalized artifact — it renders any
        # iteration of any task. Fall back to a direct read (old iterations /
        # the standalone demo) so nothing breaks when it is absent.
        rm = load_review_model(iter_dir)
        if rm is not None:
            return {**rm, "feedback": fb}
        return review_payload(model, fb)

    @app.get("/api/image")
    def image(method: str, sample: str, role: str):
        loaded = load_image(cfg, method, sample, role)
        if loaded is None:
            raise HTTPException(status_code=404, detail="image not found")
        data, media_type = loaded
        return Response(
            content=data, media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/overview")
    def overview() -> dict:
        return _overview(iter_dir)

    @app.put("/api/feedback/item")
    def put_item(body: dict = Body(...)) -> dict:
        for key in ("sample_key", "method_id", "combination_id", "label"):
            if not body.get(key):
                raise HTTPException(status_code=422, detail=f"missing {key}")
        with lock:
            fb = read_feedback(iter_dir, model.iteration, model.mode)
            set_item_mark(fb, body["sample_key"], body["method_id"],
                          body["combination_id"], body["label"],
                          body.get("comment", ""))
            write_feedback(iter_dir, fb)
        return {"status": "ok", "feedback": fb}

    @app.put("/api/feedback/comparison")
    def put_comparison(body: dict = Body(...)) -> dict:
        for key in ("sample_key", "combination_id", "label"):
            if not body.get(key):
                raise HTTPException(status_code=422, detail=f"missing {key}")
        with lock:
            fb = read_feedback(iter_dir, model.iteration, model.mode)
            set_comparison_mark(fb, body["sample_key"], body["combination_id"],
                                body["label"], body.get("comment", ""))
            write_feedback(iter_dir, fb)
        return {"status": "ok", "feedback": fb}

    @app.put("/api/feedback/general")
    def put_general(body: dict = Body(...)) -> dict:
        with lock:
            fb = read_feedback(iter_dir, model.iteration, model.mode)
            set_general(fb, body.get("text", ""))
            write_feedback(iter_dir, fb)
        return {"status": "ok", "feedback": fb}

    @app.post("/api/feedback/save")
    def save() -> dict:
        with lock:
            fb = read_feedback(iter_dir, model.iteration, model.mode)
            write_feedback(iter_dir, fb)
        return {"status": "saved", "updated_at": fb.get("updated_at")}

    @app.post("/api/feedback/finalize")
    def finalize(body: dict = Body(default={})) -> dict:
        reopen = bool((body or {}).get("reopen"))
        with lock:
            fb = read_feedback(iter_dir, model.iteration, model.mode)
            if fb.get("status") == "finalized" and not reopen:
                raise HTTPException(
                    status_code=409,
                    detail="feedback already finalized — pass reopen:true to re-finalize",
                )
            result = finalize_feedback(workspace_root, iteration=model.iteration)
        return result

    # ---- static front-end ---------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_html = STATIC_DIR / "index.html"
        if index_html.is_file():
            return FileResponse(index_html)
        return HTMLResponse(_PLACEHOLDER)

    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _overview(iter_dir: Path) -> dict:
    """Distil the cross-family comparison artifacts for the Overview view."""
    results = iter_dir / "experiments" / "results"

    def _load(path: Path) -> object:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    return {
        "summary": _load(results / "summary.json"),
        "comparison": _load(iter_dir / "comparison" / "comparison.json"),
        "final_comparison": _load(results / "full" / "final_comparison.json"),
        "diagnostics": _load(
            results / "full" / "posthoc-disagreement-analysis" / "diagnostics.json"
        ),
    }
