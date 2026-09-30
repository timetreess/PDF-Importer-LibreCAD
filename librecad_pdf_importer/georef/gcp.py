"""Versioned JSON persistence for ground-control points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

from .models import GCP, GeoreferenceError


SCHEMA_VERSION = 1
_COORDINATE_KEYS = ("source_x", "source_y", "world_x", "world_y")


def save_gcps(path: str | Path, gcps: Iterable[GCP]) -> None:
    """Write GCPs using the stable Version 1 interchange schema."""

    rows = []
    for index, gcp in enumerate(gcps, start=1):
        if not isinstance(gcp, GCP):
            raise GeoreferenceError(f"GCP {index} must be a GCP instance")
        rows.append(
            {
                "label": gcp.label,
                "source_x": gcp.source_x,
                "source_y": gcp.source_y,
                "world_x": gcp.world_x,
                "world_y": gcp.world_y,
            }
        )
    payload = {"schema_version": SCHEMA_VERSION, "gcps": rows}
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_gcps(path: str | Path) -> Tuple[GCP, ...]:
    """Load and validate Version 1 GCP JSON."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GeoreferenceError(f"Unable to read GCP JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GeoreferenceError("GCP JSON root must be an object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise GeoreferenceError(f"GCP JSON schema_version must be {SCHEMA_VERSION}")
    rows = payload.get("gcps")
    if not isinstance(rows, list):
        raise GeoreferenceError("GCP JSON gcps must be an array")

    gcps = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise GeoreferenceError(f"GCP {index} must be an object")
        missing = [key for key in _COORDINATE_KEYS if key not in row]
        if missing:
            raise GeoreferenceError(
                f"GCP {index} is missing required field(s): {', '.join(missing)}"
            )
        unknown = set(row) - {*_COORDINATE_KEYS, "label"}
        if unknown:
            raise GeoreferenceError(
                f"GCP {index} has unknown field(s): {', '.join(sorted(unknown))}"
            )
        try:
            gcps.append(
                GCP(
                    source_x=row["source_x"],
                    source_y=row["source_y"],
                    world_x=row["world_x"],
                    world_y=row["world_y"],
                    label=row.get("label", ""),
                )
            )
        except GeoreferenceError as exc:
            raise GeoreferenceError(f"GCP {index}: {exc}") from exc
    return tuple(gcps)
