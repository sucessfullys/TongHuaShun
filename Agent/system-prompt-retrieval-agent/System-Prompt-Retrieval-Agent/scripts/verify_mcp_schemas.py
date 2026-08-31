#!/usr/bin/env python3
"""
Dev-only script that validates all JSON schemas in mcp_tools/schemas/ and
checks that schemas mapping to pydantic models include the required field sets.

Usage:
    .venv/bin/python scripts/verify_mcp_schemas.py

Exit codes:
    0  all checks passed
    1  one or more checks failed (error list printed to stdout)
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema is not installed. Run: pip install jsonschema")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "mcp_tools" / "schemas"
SRC_PACKAGE = "system_prompt_retrieval_agent"


# ---------------------------------------------------------------------------
# Required-field mappings
# Schemas that must contain a superset of the required fields of a pydantic model.
# ---------------------------------------------------------------------------

# Fields that are required (non-Optional, no default) in RemoteStageRequest
REMOTE_STAGE_REQUEST_REQUIRED = {
    "run_id",
    "round_id",
    "prompt_pair_id",
    "system_prompt_id",
    "system_prompt_text",
    "dataset_root",
    "output_root",
}

# Fields that are required in RemoteStageResponse
REMOTE_STAGE_RESPONSE_REQUIRED = {
    "ok",
    "stage",
}

# Maps schema filename -> set of fields the schema's "required" array must contain
REQUIRED_FIELD_CHECKS: dict[str, set[str]] = {
    "remote_prompt_pair_pipeline.input.json": REMOTE_STAGE_REQUEST_REQUIRED,
    "remote_prompt_pair_pipeline.output.json": REMOTE_STAGE_RESPONSE_REQUIRED,
    "remote_gemma_stage.input.json": REMOTE_STAGE_REQUEST_REQUIRED,
    "remote_gemma_stage.output.json": REMOTE_STAGE_RESPONSE_REQUIRED,
    "remote_flux_stage.input.json": REMOTE_STAGE_REQUEST_REQUIRED,
    "remote_flux_stage.output.json": REMOTE_STAGE_RESPONSE_REQUIRED,
    "remote_qwen_stage.input.json": REMOTE_STAGE_REQUEST_REQUIRED,
    "remote_qwen_stage.output.json": REMOTE_STAGE_RESPONSE_REQUIRED,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_schemas() -> list[tuple[Path, dict]]:
    """Return a list of (path, schema_dict) for every JSON file in SCHEMAS_DIR."""
    if not SCHEMAS_DIR.exists():
        print(f"ERROR: schemas directory not found: {SCHEMAS_DIR}")
        sys.exit(1)
    files = sorted(SCHEMAS_DIR.glob("*.json"))
    if not files:
        print(f"ERROR: no JSON files found in {SCHEMAS_DIR}")
        sys.exit(1)
    result = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid JSON in {path.name}: {exc}")
            sys.exit(1)
        result.append((path, data))
    return result


def check_schema_validity(path: Path, schema: dict, errors: list[str]) -> None:
    """Validate that schema is a legal JSON Schema Draft 2020-12."""
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        errors.append(f"[{path.name}] invalid schema: {exc.message}")


def check_required_fields(
    path: Path,
    schema: dict,
    expected: set[str],
    errors: list[str],
) -> None:
    """Check that schema['required'] is a superset of expected."""
    actual = set(schema.get("required", []))
    missing = expected - actual
    if missing:
        errors.append(
            f"[{path.name}] missing required fields: {sorted(missing)}"
        )


def verify_pydantic_model_importable(errors: list[str]) -> bool:
    """
    Attempt to import the schemas module to confirm the package is on sys.path.
    Returns True if successful, False if the import fails (non-fatal for schema
    checks but reported as a warning).
    """
    try:
        mod = importlib.import_module(f"{SRC_PACKAGE}.schemas")
        _ = mod.RemoteStageRequest
        _ = mod.RemoteStageResponse
        return True
    except (ImportError, AttributeError) as exc:
        errors.append(
            f"[pydantic import check] WARNING: could not import {SRC_PACKAGE}.schemas: {exc}. "
            "Field-set comparison skipped."
        )
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    schemas = load_schemas()
    print(f"Found {len(schemas)} schema file(s) in {SCHEMAS_DIR}")

    # Step 1: validate each schema is legal JSON Schema Draft 2020-12
    for path, schema in schemas:
        check_schema_validity(path, schema, errors)
    print(f"  Schema validity checks: {len(schemas)} file(s) checked")

    # Step 2: verify pydantic model importability (informational)
    can_import = verify_pydantic_model_importable(warnings)
    if can_import:
        print(f"  Pydantic model import: OK ({SRC_PACKAGE}.schemas)")
    else:
        print(f"  Pydantic model import: WARNING (see below)")

    # Step 3: required-field superset checks
    schema_map: dict[str, dict] = {path.name: schema for path, schema in schemas}
    for filename, expected_fields in REQUIRED_FIELD_CHECKS.items():
        if filename not in schema_map:
            errors.append(f"[{filename}] file not found in schemas directory")
            continue
        check_required_fields(
            Path(filename),
            schema_map[filename],
            expected_fields,
            errors,
        )
    print(f"  Required-field superset checks: {len(REQUIRED_FIELD_CHECKS)} mapping(s) checked")

    # Report
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print(f"\nFAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
