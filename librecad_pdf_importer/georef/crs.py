"""Output CRS validation without coordinate reprojection."""
from __future__ import annotations

import math

from pyproj import CRS
from pyproj.exceptions import CRSError

from .models import CRSDefinition, GeoreferenceError


def validate_crs(value: str | int | None = None) -> CRSDefinition:
    """Validate local mode or return a canonical EPSG authority definition."""

    if value is None or (isinstance(value, str) and value.strip().lower() == "local"):
        return CRSDefinition(mode="local", authority=None, name="Local coordinates")
    if isinstance(value, bool):
        raise GeoreferenceError("Output CRS must be local or a valid EPSG CRS")
    try:
        crs = CRS.from_user_input(value)
    except (CRSError, TypeError, ValueError) as exc:
        raise GeoreferenceError(f"Invalid EPSG CRS: {value!r}") from exc
    authority = crs.to_authority()
    if authority is None or authority[0].upper() != "EPSG":
        raise GeoreferenceError(f"Output CRS must resolve to an EPSG authority: {value!r}")
    axes = tuple(crs.axis_info)
    metre_axes = all(
        str(axis.unit_name).strip().lower() in {"metre", "meter"}
        and math.isclose(
            float(axis.unit_conversion_factor),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for axis in axes
    )
    if not crs.is_projected or len(axes) != 2 or not metre_axes:
        raise GeoreferenceError(
            f"EPSG CRS must be a projected 2D CRS with metre horizontal axes: {value!r}"
        )
    return CRSDefinition(
        mode="epsg",
        authority=f"EPSG:{int(authority[1])}",
        name=crs.name,
    )
