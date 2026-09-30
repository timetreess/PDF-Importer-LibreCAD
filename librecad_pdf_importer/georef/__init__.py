"""Manual-GCP georeferencing primitives for PDF2GeoCAD."""

from .affine import solve_affine
from .crs import validate_crs
from .gcp import load_gcps, save_gcps
from .helmert import solve_helmert
from .models import (
    AffineCoefficients,
    CRSDefinition,
    GCP,
    GCPResidual,
    GeoreferenceError,
    HelmertCoefficients,
    ResidualSummary,
    TransformResult,
)
from .residual import calculate_residuals
from .transform_geometry import transform_extraction

__all__ = [
    "AffineCoefficients",
    "CRSDefinition",
    "GCP",
    "GCPResidual",
    "GeoreferenceError",
    "HelmertCoefficients",
    "ResidualSummary",
    "TransformResult",
    "calculate_residuals",
    "load_gcps",
    "save_gcps",
    "solve_affine",
    "solve_helmert",
    "transform_extraction",
    "validate_crs",
]
