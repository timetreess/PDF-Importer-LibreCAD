"""Rank-checked least-squares 2D Helmert solver."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import GCP, GeoreferenceError, HelmertCoefficients, TransformResult
from .residual import calculate_residuals


def solve_helmert(gcps: Iterable[GCP]) -> TransformResult:
    """Fit ``X=a*x-b*y+tx, Y=b*x+a*y+ty`` to two or more GCPs."""

    points = tuple(gcps)
    if len(points) < 2:
        raise GeoreferenceError("Helmert transform requires at least 2 GCPs")
    if not all(isinstance(gcp, GCP) for gcp in points):
        raise GeoreferenceError("Helmert transform requires GCP instances")
    if len({gcp.source for gcp in points}) != len(points):
        raise GeoreferenceError(
            "Helmert source GCP layout is rank-deficient because it contains a duplicate point"
        )
    if len({gcp.world for gcp in points}) < 2:
        raise GeoreferenceError("Helmert destination GCP layout is collapsed")

    design = np.empty((2 * len(points), 4), dtype=float)
    observations = np.empty(2 * len(points), dtype=float)
    for index, gcp in enumerate(points):
        row = 2 * index
        design[row] = (gcp.source_x, -gcp.source_y, 1.0, 0.0)
        design[row + 1] = (gcp.source_y, gcp.source_x, 0.0, 1.0)
        observations[row : row + 2] = gcp.world

    try:
        solution, _squared_error, rank, _singular_values = np.linalg.lstsq(
            design, observations, rcond=None
        )
    except np.linalg.LinAlgError as exc:
        raise GeoreferenceError("Helmert transform could not be solved") from exc
    if rank < 4:
        raise GeoreferenceError(
            "Helmert source GCP layout is rank-deficient; use distinct source points"
        )
    if not np.isfinite(solution).all():
        raise GeoreferenceError("Helmert solved mapping must contain finite coefficients")

    coefficients = HelmertCoefficients(*map(float, solution))
    scale = math.hypot(coefficients.a, coefficients.b)
    if not math.isfinite(scale):
        raise GeoreferenceError("Helmert solved mapping must contain finite coefficients")
    if scale == 0.0:
        raise GeoreferenceError("Helmert solved mapping must be invertible")
    try:
        summary = calculate_residuals(points, coefficients)
    except (FloatingPointError, OverflowError) as exc:
        raise GeoreferenceError("Helmert solved mapping must produce finite residuals") from exc
    if not math.isfinite(summary.rmse) or not math.isfinite(summary.max_residual):
        raise GeoreferenceError("Helmert solved mapping must produce finite residuals")
    return TransformResult(
        method="helmert",
        transform_coefficients=coefficients,
        residuals=summary.residuals,
        rmse=summary.rmse,
        max_residual=summary.max_residual,
        scale=scale,
        rotation_deg=math.degrees(math.atan2(coefficients.b, coefficients.a)),
    )
