"""FastAPI routes for the image-annotation web app.

``build_app(dataset_root)`` returns an app bound to one dataset. The
:class:`Dataset` is walked once at app construction (via the Stage 0 probe)
and reused for every request, so per-request latency is just an in-memory
lookup plus the LRU-cached image read.

Routes accept ``{sample_key:path}`` so slash-bearing sample keys
(``dress/dress_complex_background/sample_001``) route correctly. The
``PUT /api/sample/{sample_key}/annotation`` body carries a single
``method_id`` + ``annotation`` text — concurrent edits to different
methods of the same sample are independent.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .data import apply_overrides, walk_dataset
from .images import cache_stats, load_image
from .store import (
    annotated_method_count,
    has_any_annotation,
    mirror_central_to_per_method,
    read_annotation,
    write_per_method_annotation,
)

STATIC_DIR = Path(__file__).parent / "static"

_PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8">
<title>ERA — Image Annotation</title></head><body style="font-family:sans-serif;
max-width:40rem;margin:4rem auto;line-height:1.6">
<h1>ERA — Image Annotation</h1>
<p>The annotation front-end is missing. The API is live; see
<a href="/api/samples">/api/samples</a>.</p>
</body></html>"""


def build_app(
    dataset_root: str | Path,
    *,
    output_overrides: dict[str, str] | None = None,
    input_role_overrides: dict[str, str] | None = None,
) -> FastAPI:
    """Build the FastAPI app for one dataset.

    Optional ``output_overrides`` / ``input_role_overrides`` (operator
    confirmations from the slash command when the probe was ambiguous)
    are applied after :func:`walk_dataset` so :meth:`Dataset.image_path_for`
    resolves to the operator's choices.
    """
    dataset = walk_dataset(Path(dataset_root))
    apply_overrides(
        dataset,
        output_overrides=output_overrides,
        input_role_overrides=input_role_overrides,
    )
    # Backfill per-method copies once at startup so any annotations the
    # operator already saved (or any out-of-band edits to the central
    # tree) land next to the corresponding output images. Idempotent;
    # only writes when a copy differs from the central source of truth.
    try:
        mirror_central_to_per_method(dataset.root, dataset.method_paths)
    except OSError:
        # Best-effort — never block the server from coming up.
        pass
    lock = threading.Lock()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="ERA Image Annotation", docs_url="/api/docs")

    # ---- API ----------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "dataset_root": str(dataset.root),
            "method_count": len(dataset.methods),
            "sample_count": len(dataset.sample_keys),
            "input_roles": list(dataset.input_roles),
            "output_role": dataset.output_role,
            "warnings": list(dataset.warnings),
            "cache": cache_stats(),
        }

    @app.get("/api/samples")
    def samples() -> dict:
        rows = [
            {"sample_key": sk,
             "annotated": has_any_annotation(dataset.root, sk),
             "annotated_methods": annotated_method_count(dataset.root, sk)}
            for sk in dataset.sample_keys
        ]
        return {
            "dataset_root": str(dataset.root),
            "methods": list(dataset.methods),
            "input_roles": list(dataset.input_roles),
            "output_role": dataset.output_role,
            "all_roles": dataset.all_roles,
            "samples": rows,
            "annotated_count": sum(1 for r in rows if r["annotated"]),
            "total_count": len(rows),
        }

    @app.get("/api/sample/{sample_key:path}")
    def sample(sample_key: str) -> dict:
        if sample_key not in dataset.sample_methods:
            raise HTTPException(status_code=404, detail="sample not found")
        annotation = read_annotation(dataset.root, sample_key)
        methods_present = dataset.methods_for_sample(sample_key)
        method_payloads = []
        for method_id in dataset.methods:
            present = method_id in methods_present
            roles_map: dict[str, str | None] = {}
            if present:
                for role in dataset.all_roles:
                    p = dataset.image_path_for(method_id, sample_key, role)
                    if p is not None:
                        try:
                            roles_map[role] = str(p.relative_to(dataset.root))
                        except ValueError:
                            roles_map[role] = None
                    else:
                        roles_map[role] = None
            else:
                roles_map = {role: None for role in dataset.all_roles}
            method_payloads.append({
                "method_id": method_id,
                "present": present,
                "roles": roles_map,
            })
        return {
            "sample_key": sample_key,
            "methods": method_payloads,
            "per_method": annotation.get("per_method", {}),
            "created_at": annotation.get("created_at"),
            "updated_at": annotation.get("updated_at"),
        }

    @app.get("/api/image")
    def image(method: str, sample: str, role: str):
        loaded = load_image(dataset, method, sample, role)
        if loaded is None:
            raise HTTPException(status_code=404, detail="image not found")
        data, media_type = loaded
        return Response(
            content=data, media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.put("/api/sample/{sample_key:path}/annotation")
    def put_annotation(sample_key: str, body: dict = Body(...)) -> dict:
        if sample_key not in dataset.sample_methods:
            raise HTTPException(status_code=404, detail="sample not found")
        method_id = (body.get("method_id") or "").strip()
        if not method_id:
            raise HTTPException(status_code=422,
                                detail="missing 'method_id' field")
        if method_id not in dataset.methods:
            raise HTTPException(status_code=422,
                                detail=f"unknown method_id: {method_id!r}")
        text = body.get("annotation")
        if text is None:
            raise HTTPException(status_code=422,
                                detail="missing 'annotation' field")
        with lock:
            try:
                record = write_per_method_annotation(
                    dataset.root, sample_key, method_id, text,
                    method_paths=dataset.method_paths,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
        return {"status": "ok", **record}

    # ---- static front-end ---------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_html = STATIC_DIR / "index.html"
        if index_html.is_file():
            return FileResponse(index_html)
        return HTMLResponse(_PLACEHOLDER)

    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
    return app
