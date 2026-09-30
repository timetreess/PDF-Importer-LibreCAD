"""Rank-checked least-squares 2D affine solver."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import AffineCoefficients, GCP, GeoreferenceError, TransformResult
from .residual import calculate_residuals


def solve_affine(gcps: Iterable[GCP]) -> TransformResult:
    """Fit a six-coefficient affine transform to three or more GCPs."""

    points = tuple(gcps)
    if len(points) < 3:
        raise GeoreferenceError("Affine transform requires at least 3 GCPs")
    if not all(isinstance(gcp, GCP) for gcp in points):
        raise GeoreferenceError("Affine transform requires GCP instances")
    if len({gcp.source for gcp in points}) != len(points):
        raise GeoreferenceError(
            "Affine source GCP layout is rank-deficient because it contains a duplicate point"
        )
    destination = np.asarray([gcp.world for gcp in points], dtype=float)
    destination_scale = float(np.max(np.abs(destination)))
    if destination_scale == 0.0:
        raise GeoreferenceError("Affine destination GCP layout is collapsed")
    normalized_destination = destination / destination_scale
    if np.linalg.matrix_rank(normalized_destination - normalized_destination[0]) < 2:
        raise GeoreferenceError(
            "Affine destination GCP layout is rank-deficient; use non-collinear destination points"
        )

    design = np.empty((2 * len(points), 6), dtype=float)
    observations = np.empty(2 * len(points), dtype=float)
    for index, gcp in enumerate(points):
        row = 2 * index
        design[row] = (gcp.source_x, gcp.source_y, 1.0, 0.0, 0.0, 0.0)
        design[row + 1] = (0.0, 0.0, 0.0, gcp.source_x, gcp.source_y, 1.0)
        observations[row : row + 2] = gcp.world

    try:
        solution, _squared_error, rank, _singular_values = np.linalg.lstsq(
            design, observations, rcond=None
        )
    except np.linalg.LinAlgError as exc:
        raise GeoreferenceError("Affine transform could not be solved") from exc
    if rank < 6:
        raise GeoreferenceError(
            "Affine source GCP layout is rank-deficient; use non-collinear source points"
        )
    if not np.isfinite(solution).all():
        raise GeoreferenceError("Affine solved mapping must contain finite coefficients")

    coefficients = AffineCoefficients(*map(float, solution))
    linear_scale = max(
        abs(coefficients.a),
        abs(coefficients.b),
        abs(coefficients.d),
        abs(coefficients.e),
    )
    if linear_scale == 0.0:
        raise GeoreferenceError("Affine solved mapping must be invertible")
    normalized_determinant = (
        coefficients.a / linear_scale * coefficients.e / linear_scale
        - coefficients.b / linear_scale * coefficients.d / linear_scale
    )
    if not math.isfinite(normalized_determinant):
        raise GeoreferenceError("Affine solved mapping must contain finite coefficients")
    if normalized_determinant == 0.0:
        raise GeoreferenceError("Affine solved mapping must be invertible")
    try:
        summary = calculate_residuals(points, coefficients)
    except (FloatingPointError, OverflowError) as exc:
        raise GeoreferenceError("Affine solved mapping must produce finite residuals") from exc
    if not math.isfinite(summary.rmse) or not math.isfinite(summary.max_residual):
        raise GeoreferenceError("Affine solved mapping must produce finite residuals")
    return TransformResult(
        method="affine",
        transform_coefficients=coefficients,
        residuals=summary.residuals,
        rmse=summary.rmse,
        max_residual=summary.max_residual,
    )
