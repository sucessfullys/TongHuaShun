"""ERA image-annotation web app — standalone (no workspace dependency).

Mirrors the structure of ``era.webapp`` (the Stage 8 review app) but operates
directly against a dataset on disk, not against an ERA iteration's results.
Launched by the ``/era:annotate`` slash command through
:mod:`era.orchestration.annotate`.
"""

from __future__ import annotations
