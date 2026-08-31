"""ERA Stage 8 human-feedback web app.

The FastAPI app lives in :mod:`era.webapp.app` (``build_app``); the detached
server entry point is :mod:`era.webapp.server`. Importing those pulls in
FastAPI/uvicorn — this package's ``data`` / ``store`` / ``images`` modules
deliberately do **not**, so the orchestration layer can reuse them without the
web stack. This ``__init__`` stays import-light for the same reason.
"""
