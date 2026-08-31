"""Walk a try-on / generation dataset using the Stage 0 probe for layout discovery.

The Stage 0 probe (:func:`era.probe.data.probe_data_roots`) already knows how
to discover the methods, ``sample_glob``, per-method ``output_file``, and
``input_roles`` of an arbitrary dataset (depths up to 8, mixed input filenames,
intermediate-artifact skipping, etc.). The annotation app delegates to it
instead of hand-rolling a walker — so any real dataset that Stage 0 can
ingest, this app can annotate.

Returned :class:`Dataset` exposes:

- ``methods`` (sorted method ids — normalized via the probe's ``_ident``);
- ``sample_keys`` (sorted union of per-method relpaths from each method dir
  to its sample dirs; same key shared across methods when present in both);
- ``input_roles`` (role id → input filename, e.g. ``input_cloth →
  input_cloth.png``);
- ``method_outputs`` (method id → that method's output filename — different
  methods can carry different filenames, e.g. ``tryon_result_flux2klein.png``
  vs. ``tryon_result_pe.png``);
- :meth:`Dataset.image_path_for` — the single resolver the FastAPI image
  route uses, with a path-traversal guard.
- ``probe_summary`` — the full probe output for diagnostics.
- ``warnings`` — operator-visible notes (mirrors the probe's ``notes`` plus
  any walker-specific issues).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..probe.data import probe_data_roots

# Canonical role name for each method's generated output (the role the
# probe doesn't name explicitly — it just gives the filename via
# ``method["output_file"]``). Input roles come from the probe's
# ``input_roles`` dict and may include arbitrary role ids
# (``input_cloth``, ``input_model``, ``source``, ``mask`` …).
OUTPUT_ROLE = "output"

# Image extensions recognized by the flat-layout detector. Mirrors the
# media-type table in ``images.py`` (kept in sync by intent — both are the
# set of files the annotation app knows how to serve).
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
}

# The single synthetic method id used for a flat-image dataset (a directory
# of standalone images, optionally bucketed into one level of subdirs such
# as ``good/`` and ``bad/``). Each image is its own sample; there is no
# per-method comparison, so one column suffices. Annotations land in
# ``per_method["image"]`` of the central record.
FLAT_METHOD_ID = "image"

# Subdirectory the annotate app writes its own annotation files into — never
# treated as a flat-image bucket.
_ANNOTATIONS_DIRNAME = "annotations"


@dataclass
class Dataset:
    """Walked state of one annotation dataset."""
    root: Path
    methods: list[str]              # sorted method ids
    sample_keys: list[str]          # sorted union across methods
    method_paths: dict[str, Path]   # method id → on-disk method dir
    method_outputs: dict[str, str]  # method id → output filename (per-method)
    input_roles: dict[str, str]     # role id → input filename
    sample_methods: dict[str, set[str]] = field(default_factory=dict)
    probe_summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Flat-image mode only: sample_key → absolute image path. Empty for the
    # normal ``<method>/<sample_dir>/<role_images>`` layout. When non-empty,
    # :meth:`image_path_for` resolves the OUTPUT_ROLE from this map directly
    # (the image *is* the sample, not a file inside a sample directory).
    flat_files: dict[str, Path] = field(default_factory=dict)

    # ---- query helpers ---------------------------------------------

    @property
    def output_role(self) -> str:
        """The canonical name of the per-method output role."""
        return OUTPUT_ROLE

    @property
    def all_roles(self) -> list[str]:
        """Every role the dataset exposes: inputs (sorted) then output."""
        return sorted(self.input_roles) + [OUTPUT_ROLE]

    def methods_for_sample(self, sample_key: str) -> list[str]:
        """Methods that actually contain this sample, sorted."""
        return sorted(self.sample_methods.get(sample_key, set()))

    def image_path_for(
        self, method_id: str, sample_key: str, role: str,
    ) -> Path | None:
        """Resolve one image path with a path-traversal guard.

        Returns ``None`` when:
        - the method is unknown,
        - the role is unknown for this method,
        - the resolved path escapes the method dir (sample_key traversal),
        - or the file is missing on disk.
        """
        # Flat-image mode: the image *is* the sample. Only the output role
        # exists; resolve it straight from the pre-built map with a
        # traversal guard against root.
        if self.flat_files:
            if role != OUTPUT_ROLE:
                return None
            candidate = self.flat_files.get(sample_key)
            if candidate is None:
                return None
            root_resolved = self.root.resolve()
            resolved = candidate.resolve()
            if (root_resolved != resolved
                    and root_resolved not in resolved.parents):
                return None
            return resolved if resolved.is_file() else None

        method_path = self.method_paths.get(method_id)
        if method_path is None:
            return None
        if role == OUTPUT_ROLE:
            filename = self.method_outputs.get(method_id, "")
        else:
            filename = self.input_roles.get(role, "")
        if not filename:
            return None

        method_path_resolved = method_path.resolve()
        candidate = (method_path / sample_key / filename).resolve()
        # Path-traversal guard: the resolved file must sit strictly under
        # the method dir. Defends against a hand-crafted sample_key like
        # "../../etc/passwd" arriving via an HTTP query.
        if (method_path_resolved != candidate
                and method_path_resolved not in candidate.parents):
            return None
        return candidate if candidate.is_file() else None


def apply_overrides(
    dataset: Dataset,
    *,
    output_overrides: dict[str, str] | None = None,
    input_role_overrides: dict[str, str] | None = None,
) -> Dataset:
    """Apply operator-confirmed overrides to a walked dataset (in place).

    Used when the Stage 0 probe returned ``confidence: needs_confirmation``
    for a method's output file (multiple candidates) or for missing input
    roles, and the slash command asked the operator to disambiguate. The
    overrides flow through env vars to the spawned server and land here
    so :meth:`Dataset.image_path_for` resolves to the operator's choices.

    - ``output_overrides`` maps ``method_id → filename`` (replaces or sets
      ``dataset.method_outputs[method_id]``). Unknown method ids are
      silently ignored (the operator's UI never offers them, but a stale
      override shouldn't crash).
    - ``input_role_overrides`` maps ``role_id → filename`` (replaces or
      adds to ``dataset.input_roles``).

    The on-disk path-traversal guard still applies on every resolution
    via :meth:`Dataset.image_path_for`, so an override like
    ``"../../etc/passwd"`` still gets rejected at fetch time — we don't
    have to re-validate filenames here.

    Returns the same dataset (mutated) for call-site chaining.
    """
    for method_id, filename in (output_overrides or {}).items():
        if method_id in dataset.method_paths:
            dataset.method_outputs[method_id] = filename
    for role_id, filename in (input_role_overrides or {}).items():
        dataset.input_roles[role_id] = filename
    return dataset


def _list_images(d: Path) -> list[Path]:
    """Sorted, non-hidden image files directly inside ``d`` (no recursion)."""
    try:
        entries = list(d.iterdir())
    except OSError:
        return []
    return sorted(
        p for p in entries
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTS
    )


def _has_subdirs(d: Path) -> bool:
    """True if ``d`` contains any non-hidden subdirectory."""
    try:
        return any(
            p.is_dir() and not p.name.startswith(".") for p in d.iterdir()
        )
    except OSError:
        return False


def detect_flat_layout(root: Path) -> dict[str, list[Path]] | None:
    """Detect a flat-image dataset; return ``{bucket: [image, ...]}`` or None.

    A flat dataset is a directory of standalone images — not the
    ``<method>/<sample_dir>/<role_images>`` shape the Stage 0 probe expects.
    Two flat shapes are recognized:

    - **Single bucket** — images sit directly under ``root`` (no meaningful
      subdirectories). The one bucket is keyed ``""``.
    - **Bucketed** — ``root`` holds one level of subdirectories (e.g.
      ``good/`` and ``bad/``), each containing image files and *no* further
      subdirectories. Each subdir is one bucket keyed by its name.

    The rule is deliberately conservative so it never misfires on a real
    multi-method dataset: there every ``<method>/`` dir contains sample
    *directories*, so :func:`_has_subdirs` is true and detection bails to
    ``None`` (the probe path). The app's own ``annotations/`` subdir and any
    hidden entries are ignored at every level.
    """
    try:
        children = [p for p in root.iterdir() if not p.name.startswith(".")]
    except OSError:
        return None
    child_dirs = [
        p for p in children
        if p.is_dir() and p.name != _ANNOTATIONS_DIRNAME
    ]
    root_imgs = _list_images(root)

    # Single-bucket: images directly under root, nothing deeper to walk.
    if root_imgs and not child_dirs:
        return {"": root_imgs}

    # Bucketed: every candidate subdir must be a leaf bucket of images.
    if child_dirs and not root_imgs:
        buckets: dict[str, list[Path]] = {}
        for d in child_dirs:
            if _has_subdirs(d):
                # A nested directory means this is a normal layered layout,
                # not a flat image collection — defer to the probe.
                return None
            imgs = _list_images(d)
            if imgs:
                buckets[d.name] = imgs
        return buckets or None

    return None


def _walk_flat_dataset(
    root_resolved: Path, buckets: dict[str, list[Path]],
) -> Dataset:
    """Build a :class:`Dataset` for a detected flat-image collection.

    Each image becomes a sample whose key is ``<bucket>/<filename>`` (or just
    ``<filename>`` for the single unbucketed bucket). One synthetic method,
    :data:`FLAT_METHOD_ID`, carries the lone output column; there are no
    input roles. The bucket name (e.g. ``good`` / ``bad``) stays visible in
    the sample_key, so the operator's existing labelling is preserved and
    Stage 7 / Stage 9 can recover it from the annotation path.
    """
    flat_files: dict[str, Path] = {}
    sample_methods: dict[str, set[str]] = {}
    for bucket in sorted(buckets):
        for img in buckets[bucket]:
            sample_key = f"{bucket}/{img.name}" if bucket else img.name
            flat_files[sample_key] = img.resolve()
            sample_methods[sample_key] = {FLAT_METHOD_ID}

    sample_keys = sorted(flat_files)
    bucket_names = [b for b in sorted(buckets) if b] or None
    total = len(sample_keys)
    if bucket_names:
        note = (
            f"flat-image layout: {total} images across "
            f"{len(bucket_names)} bucket(s) ({', '.join(bucket_names)}). "
            f"Each image is one sample; annotate it in the single "
            f"'{FLAT_METHOD_ID}' column."
        )
    else:
        note = (
            f"flat-image layout: {total} images directly under the root. "
            f"Each image is one sample; annotate it in the single "
            f"'{FLAT_METHOD_ID}' column."
        )

    probe_summary = {
        "layout": "flat_files",
        "confidence": "high",
        "sample_glob": "*",
        "methods": [{
            "method_id": FLAT_METHOD_ID,
            "path": str(root_resolved),
            "output_file": "",
            "output_candidates": [],
        }],
        "input_roles": {},
        "notes": [note],
    }

    return Dataset(
        root=root_resolved,
        methods=[FLAT_METHOD_ID],
        sample_keys=sample_keys,
        # method_path = root so the (best-effort) per-method mirror writer
        # resolves under root and then safely skips — a flat sample_key
        # names a file, not a directory, so no mirror copy is ever written.
        method_paths={FLAT_METHOD_ID: root_resolved},
        method_outputs={FLAT_METHOD_ID: ""},
        input_roles={},
        sample_methods=sample_methods,
        probe_summary=probe_summary,
        warnings=[note],
        flat_files=flat_files,
    )


def walk_dataset(root: Path) -> Dataset:
    """Walk ``root`` via the Stage 0 probe; return a :class:`Dataset`.

    Raises :class:`FileNotFoundError` when ``root`` is not a directory.
    Anything past that — empty methods, ambiguous layout, missing
    output files for some methods — is reported via
    :attr:`Dataset.warnings`, not raised, so the web app can still come
    up and let the operator triage.

    A **flat-image collection** (a directory of standalone images, optionally
    bucketed one level deep — see :func:`detect_flat_layout`) is handled
    before the probe: each image becomes a single-column sample. This is the
    ``/era:annotate`` per-image-notes mode for datasets that are not the
    multi-method ``<method>/<sample>/<images>`` shape.
    """
    root_resolved = Path(root).resolve()
    if not root_resolved.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root_resolved}")

    flat_buckets = detect_flat_layout(root_resolved)
    if flat_buckets:
        return _walk_flat_dataset(root_resolved, flat_buckets)

    probe = probe_data_roots([str(root_resolved)])
    warnings: list[str] = list(probe.get("notes") or [])

    methods_raw = probe.get("methods") or []
    sample_glob = probe.get("sample_glob") or ""
    input_roles = dict(probe.get("input_roles") or {})

    method_paths: dict[str, Path] = {}
    method_outputs: dict[str, str] = {}
    sample_methods: dict[str, set[str]] = {}

    for entry in methods_raw:
        method_id = entry.get("method_id") or ""
        method_path = entry.get("path") or ""
        output_file = entry.get("output_file") or ""
        if not method_id or not method_path:
            continue
        if not output_file:
            warnings.append(
                f"method {method_id!r} has no detected output_file; "
                f"its column will show 'missing' in the UI"
            )
        m_dir = Path(method_path)
        method_paths[method_id] = m_dir
        method_outputs[method_id] = output_file

        # Enumerate this method's sample dirs via the probe's sample_glob.
        # Fall back to a single-level wildcard if the probe didn't infer one.
        glob_pat = sample_glob or "*"
        for sample_dir in sorted(m_dir.glob(glob_pat)):
            if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                continue
            try:
                rel = sample_dir.resolve().relative_to(m_dir.resolve())
            except ValueError:
                continue
            # Use POSIX-style forward slashes for sample_key (stable across
            # platforms and matches Stage 8's convention).
            sample_key = rel.as_posix()
            sample_methods.setdefault(sample_key, set()).add(method_id)

    methods = sorted(method_paths)
    sample_keys = sorted(sample_methods)

    return Dataset(
        root=root_resolved,
        methods=methods,
        sample_keys=sample_keys,
        method_paths=method_paths,
        method_outputs=method_outputs,
        input_roles=input_roles,
        sample_methods=sample_methods,
        probe_summary=probe,
        warnings=warnings,
    )
