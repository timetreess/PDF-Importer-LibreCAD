"""Residual and quality-control calculations for solved transforms."""
from __future__ import annotations

import math
from typing import Iterable

from .models import GCP, GCPResidual, GeoreferenceError, ResidualSummary


def calculate_residuals(gcps: Iterable[GCP], transform) -> ResidualSummary:
    """Return computed-minus-control residuals and radial error statistics."""

    points = tuple(gcps)
    if not points:
        raise GeoreferenceError("At least one GCP is required to calculate residuals")
    if not hasattr(transform, "transform_point"):
        raise GeoreferenceError("Transform must provide transform_point(x, y)")

    residuals = []
    for index, gcp in enumerate(points, start=1):
        if not isinstance(gcp, GCP):
            raise GeoreferenceError(f"GCP {index} must be a GCP instance")
        predicted_x, predicted_y = transform.transform_point(*gcp.source)
        residual_x = float(predicted_x - gcp.world_x)
        residual_y = float(predicted_y - gcp.world_y)
        total_error = math.hypot(residual_x, residual_y)
        residuals.append(
            GCPResidual(
                gcp_index=index,
                gcp=gcp,
                residual_x=residual_x,
                residual_y=residual_y,
                total_error=total_error,
            )
        )

    rmse = math.sqrt(math.fsum(item.total_error**2 for item in residuals) / len(residuals))
    return ResidualSummary(
        residuals=tuple(residuals),
        rmse=rmse,
        max_residual=max(item.total_error for item in residuals),
    )
