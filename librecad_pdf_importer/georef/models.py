"""Immutable data models shared by the georeferencing core."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union


Point = Tuple[float, float]


class GeoreferenceError(ValueError):
    """Raised when georeferencing input cannot produce a valid result."""


def _finite_coordinate(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise GeoreferenceError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeoreferenceError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise GeoreferenceError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class GCP:
    """A source-model point paired with its known world coordinate."""

    source_x: float
    source_y: float
    world_x: float
    world_y: float
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("source_x", "source_y", "world_x", "world_y"):
            object.__setattr__(self, name, _finite_coordinate(getattr(self, name), name))
        if not isinstance(self.label, str):
            raise GeoreferenceError("GCP label must be a string")

    @property
    def source(self) -> Point:
        return self.source_x, self.source_y

    @property
    def world(self) -> Point:
        return self.world_x, self.world_y


@dataclass(frozen=True)
class CRSDefinition:
    """Validated output-coordinate metadata; no reprojection is implied."""

    mode: str
    authority: Optional[str]
    name: str

    @property
    def is_local(self) -> bool:
        return self.mode == "local"


@dataclass(frozen=True)
class HelmertCoefficients:
    a: float
    b: float
    tx: float
    ty: float

    def transform_point(self, x: float, y: float) -> Point:
        return self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty

    @property
    def linear(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.a, -self.b), (self.b, self.a))


@dataclass(frozen=True)
class AffineCoefficients:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def transform_point(self, x: float, y: float) -> Point:
        return self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f

    @property
    def linear(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.a, self.b), (self.d, self.e))


TransformCoefficients = Union[HelmertCoefficients, AffineCoefficients]


@dataclass(frozen=True)
class GCPResidual:
    gcp_index: int
    gcp: GCP
    residual_x: float
    residual_y: float
    total_error: float

    @property
    def dx(self) -> float:
        return self.residual_x

    @property
    def dy(self) -> float:
        return self.residual_y

    @property
    def error(self) -> float:
        return self.total_error


@dataclass(frozen=True)
class ResidualSummary:
    residuals: Tuple[GCPResidual, ...]
    rmse: float
    max_residual: float


@dataclass(frozen=True)
class TransformResult:
    method: str
    transform_coefficients: TransformCoefficients
    residuals: Tuple[GCPResidual, ...]
    rmse: float
    max_residual: float
    scale: Optional[float] = None
    rotation_deg: Optional[float] = None

    def transform_point(self, x: float, y: float) -> Point:
        return self.transform_coefficients.transform_point(x, y)

    @property
    def coefficients(self) -> TransformCoefficients:
        return self.transform_coefficients
