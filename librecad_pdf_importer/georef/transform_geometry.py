"""Apply a solved transform to one extracted page in place."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable, Sequence

from librecad_pdf_importer.core.document import DocumentExtraction, ExtractedPage
from pdfcadcore.primitives import NormalizedText, Primitive, TextCharLayout

from .models import (
    AffineCoefficients,
    GeoreferenceError,
    HelmertCoefficients,
    Point,
    TransformResult,
)


class _Transform:
    def __init__(
        self,
        transform: TransformResult | HelmertCoefficients | AffineCoefficients,
    ) -> None:
        coefficients = (
            transform.transform_coefficients
            if isinstance(transform, TransformResult)
            else transform
        )
        if not isinstance(coefficients, (HelmertCoefficients, AffineCoefficients)):
            raise GeoreferenceError("Unsupported georeferencing transform")
        self.coefficients = coefficients
        self.linear = coefficients.linear
        self.determinant = (
            self.linear[0][0] * self.linear[1][1]
            - self.linear[0][1] * self.linear[1][0]
        )

    def point(self, point: Sequence[float]) -> Point:
        return self.coefficients.transform_point(float(point[0]), float(point[1]))

    def vector(self, vector: Sequence[float]) -> Point:
        x, y = float(vector[0]), float(vector[1])
        return (
            self.linear[0][0] * x + self.linear[0][1] * y,
            self.linear[1][0] * x + self.linear[1][1] * y,
        )

    @property
    def nominal_scale(self) -> float:
        return math.sqrt(abs(self.determinant))

    @property
    def similarity_scale(self) -> float | None:
        first = (self.linear[0][0], self.linear[1][0])
        second = (self.linear[0][1], self.linear[1][1])
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        if first_length == 0.0 or second_length == 0.0:
            return None
        if not math.isclose(first_length, second_length, rel_tol=1e-9, abs_tol=0.0):
            return None
        normalized_dot = (
            first[0] / first_length * second[0] / second_length
            + first[1] / first_length * second[1] / second_length
        )
        if not math.isclose(normalized_dot, 0.0, rel_tol=0.0, abs_tol=1e-9):
            return None
        smaller, larger = sorted((first_length, second_length))
        return smaller + (larger - smaller) / 2.0

    def direction_angle(self, angle_deg: float) -> float:
        angle = math.radians(float(angle_deg))
        vector = self.vector((math.cos(angle), math.sin(angle)))
        if math.hypot(*vector) == 0.0:
            raise GeoreferenceError("Transform collapses an angular direction")
        return math.degrees(math.atan2(vector[1], vector[0])) % 360.0


def _bbox_from_points(points: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    coordinates = tuple((float(point[0]), float(point[1])) for point in points)
    if not coordinates:
        raise GeoreferenceError("Cannot calculate bounds without points")
    return (
        min(point[0] for point in coordinates),
        min(point[1] for point in coordinates),
        max(point[0] for point in coordinates),
        max(point[1] for point in coordinates),
    )


def _bbox_corners(bbox: Sequence[float]) -> tuple[Point, Point, Point, Point]:
    x0, y0, x1, y1 = map(float, bbox)
    return (x0, y0), (x1, y0), (x1, y1), (x0, y1)


def _transform_bbox(bbox: Sequence[float], transform: _Transform):
    return _bbox_from_points(transform.point(point) for point in _bbox_corners(bbox))


def _arc_span(start_angle: float, end_angle: float) -> float:
    raw_span = float(end_angle) - float(start_angle)
    span = raw_span % 360.0
    if math.isclose(span, 0.0, abs_tol=1e-12) and not math.isclose(
        raw_span, 0.0, abs_tol=1e-12
    ):
        return 360.0
    return span


def _sample_circle(center: Point, radius: float, segments: int) -> list[Point]:
    return [
        (
            center[0] + radius * math.cos(2.0 * math.pi * index / segments),
            center[1] + radius * math.sin(2.0 * math.pi * index / segments),
        )
        for index in range(segments)
    ]


def _sample_arc(
    center: Point,
    radius: float,
    start_angle: float,
    end_angle: float,
    segments: int,
) -> list[Point]:
    span = _arc_span(start_angle, end_angle)
    return [
        (
            center[0] + radius * math.cos(math.radians(start_angle + span * index / segments)),
            center[1] + radius * math.sin(math.radians(start_angle + span * index / segments)),
        )
        for index in range(segments + 1)
    ]


def _arc_bbox(
    center: Point,
    radius: float,
    start_angle: float,
    end_angle: float,
) -> tuple[float, float, float, float]:
    span = _arc_span(start_angle, end_angle)
    candidates = [float(start_angle), float(end_angle)]
    for angle in (0.0, 90.0, 180.0, 270.0):
        if span >= 360.0 - 1e-12 or (angle - start_angle) % 360.0 <= span + 1e-12:
            candidates.append(angle)
    return _bbox_from_points(
        (
            center[0] + radius * math.cos(math.radians(angle)),
            center[1] + radius * math.sin(math.radians(angle)),
        )
        for angle in candidates
    )


def _rect_is_axis_aligned(points: Sequence[Point]) -> bool:
    if len(points) < 4:
        return False
    first = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    second = (points[2][0] - points[1][0], points[2][1] - points[1][1])
    scale = max(math.hypot(*first), math.hypot(*second), 1.0)
    tolerance = scale * 1e-9
    first_horizontal = abs(first[1]) <= tolerance and abs(first[0]) > tolerance
    first_vertical = abs(first[0]) <= tolerance and abs(first[1]) > tolerance
    second_horizontal = abs(second[1]) <= tolerance and abs(second[0]) > tolerance
    second_vertical = abs(second[0]) <= tolerance and abs(second[1]) > tolerance
    return (first_horizontal and second_vertical) or (first_vertical and second_horizontal)


def _transform_primitive(
    primitive: Primitive,
    transform: _Transform,
    curve_segments: int,
) -> None:
    original_type = primitive.type.lower()
    original_bbox = primitive.bbox
    original_points = list(primitive.points)
    original_center = primitive.center
    original_radius = primitive.radius
    original_start = primitive.start_angle
    original_end = primitive.end_angle

    if primitive.line_width is not None:
        primitive.line_width = float(primitive.line_width) * transform.nominal_scale
    if primitive.area is not None:
        primitive.area = float(primitive.area) * abs(transform.determinant)

    similarity_scale = transform.similarity_scale
    curve_metadata = (
        original_center is not None
        and original_radius is not None
        and (original_type != "arc" or (original_start is not None and original_end is not None))
    )
    if original_type in {"circle", "arc"} and similarity_scale is None:
        if curve_metadata:
            if original_type == "circle":
                source_points = _sample_circle(original_center, float(original_radius), curve_segments)
                primitive.closed = True
            else:
                source_points = _sample_arc(
                    original_center,
                    float(original_radius),
                    float(original_start),
                    float(original_end),
                    curve_segments,
                )
                primitive.closed = False
        elif original_points:
            source_points = original_points
            primitive.closed = original_type == "circle"
        else:
            raise GeoreferenceError(
                f"Primitive {primitive.id} {original_type} lacks geometry for affine sampling"
            )
        primitive.points = [transform.point(point) for point in source_points]
        primitive.type = "polyline"
        primitive.center = None
        primitive.radius = None
        primitive.start_angle = None
        primitive.end_angle = None
        primitive.bbox = _bbox_from_points(primitive.points)
        return

    primitive.points = [transform.point(point) for point in original_points]
    if original_center is not None:
        primitive.center = transform.point(original_center)
    if original_radius is not None and similarity_scale is not None:
        primitive.radius = float(original_radius) * similarity_scale
    if original_start is not None:
        primitive.start_angle = transform.direction_angle(original_start)
    if original_end is not None:
        primitive.end_angle = transform.direction_angle(original_end)
    if (
        original_type == "arc"
        and transform.determinant < 0.0
        and primitive.start_angle is not None
        and primitive.end_angle is not None
    ):
        primitive.start_angle, primitive.end_angle = primitive.end_angle, primitive.start_angle

    if original_type == "rect":
        if len(primitive.points) < 4 and original_bbox is not None:
            primitive.points = [
                transform.point(point) for point in _bbox_corners(original_bbox)
            ]
        primitive.closed = True
        if not _rect_is_axis_aligned(primitive.points):
            primitive.type = "polyline"

    if original_type == "circle" and primitive.center is not None and primitive.radius is not None:
        center_x, center_y = primitive.center
        primitive.bbox = (
            center_x - primitive.radius,
            center_y - primitive.radius,
            center_x + primitive.radius,
            center_y + primitive.radius,
        )
    elif (
        original_type == "arc"
        and primitive.center is not None
        and primitive.radius is not None
        and primitive.start_angle is not None
        and primitive.end_angle is not None
    ):
        primitive.bbox = _arc_bbox(
            primitive.center,
            primitive.radius,
            primitive.start_angle,
            primitive.end_angle,
        )
    elif original_bbox is not None:
        primitive.bbox = _transform_bbox(original_bbox, transform)
    elif primitive.points:
        primitive.bbox = _bbox_from_points(primitive.points)


def _unit_vector(vector: Point, fallback: Point) -> Point:
    length = math.hypot(*vector)
    if length <= 1e-15:
        return fallback
    return vector[0] / length, vector[1] / length


def _character_scales(character: TextCharLayout, transform: _Transform) -> tuple[float, float]:
    quad = tuple(character.target_quad)
    if len(quad) >= 4:
        baseline = _unit_vector(
            (quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]),
            (1.0, 0.0),
        )
        vertical = _unit_vector(
            (quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]),
            (0.0, 1.0),
        )
    else:
        baseline, vertical = (1.0, 0.0), (0.0, 1.0)
    return math.hypot(*transform.vector(baseline)), math.hypot(*transform.vector(vertical))


def _transform_character(character: TextCharLayout, transform: _Transform) -> TextCharLayout:
    baseline_scale, vertical_scale = _character_scales(character, transform)
    return replace(
        character,
        target_origin=transform.point(character.target_origin),
        target_quad=tuple(transform.point(point) for point in character.target_quad),
        advance_width=float(character.advance_width) * baseline_scale,
        glyph_height=float(character.glyph_height) * vertical_scale,
    )


def _transform_text(text: NormalizedText, transform: _Transform) -> None:
    angle = math.radians(float(text.rotation))
    baseline = (math.cos(angle), math.sin(angle))
    vertical = (-math.sin(angle), math.cos(angle))
    transformed_baseline = transform.vector(baseline)
    transformed_vertical = transform.vector(vertical)
    baseline_scale = math.hypot(*transformed_baseline)
    vertical_scale = math.hypot(*transformed_vertical)
    if baseline_scale <= 1e-15 or vertical_scale <= 1e-15:
        raise GeoreferenceError(f"Text item {text.id} is collapsed by the transform")

    text.insertion = transform.point(text.insertion)
    if text.bbox is not None:
        text.bbox = _transform_bbox(text.bbox, transform)
    text.rotation = math.degrees(
        math.atan2(transformed_baseline[1], transformed_baseline[0])
    )
    text.font_size = float(text.font_size) * transform.nominal_scale
    text.advance_width = float(text.advance_width) * baseline_scale
    text.glyph_height = float(text.glyph_height) * vertical_scale
    text.baseline_descent = float(text.baseline_descent) * vertical_scale
    if text.target_quad_model is not None:
        text.target_quad_model = tuple(
            transform.point(point) for point in text.target_quad_model
        )
    text.source_char_layout = tuple(
        _transform_character(character, transform) for character in text.source_char_layout
    )


def _disable_source_overlays(page: ExtractedPage) -> None:
    page.images.clear()
    page.page_data.xobject_names.clear()
    page.image_paint_order = None
    page.source_line_dashes.clear()
    page.final_rect_paints.clear()
    page.source_capsules.clear()
    page.nontext_composites.clear()
    page.capsule_paint_order = None
    page.capsule_vector_paint_order = None
    page.display_to_model = None
    page.resolved_mode = "vector"
    page.resolved_reason = "georeferenced editable geometry"
    page.raster_fallback_failed = False


def transform_extraction(
    extraction: DocumentExtraction,
    transform: TransformResult | HelmertCoefficients | AffineCoefficients,
    *,
    curve_segments: int = 64,
) -> DocumentExtraction:
    """Transform one extracted page to world coordinates and disable unsafe overlays."""

    if len(extraction.pages) != 1:
        raise GeoreferenceError("Georeferencing requires exactly one extracted page")
    if isinstance(curve_segments, bool) or not isinstance(curve_segments, int) or curve_segments < 4:
        raise GeoreferenceError("curve_segments must be an integer of at least 4")

    operation = _Transform(transform)
    page = extraction.pages[0]
    page_data = page.page_data
    source_page_corners = (
        (0.0, 0.0),
        (float(page_data.width), 0.0),
        (float(page_data.width), float(page_data.height)),
        (0.0, float(page_data.height)),
    )
    page_bounds = _bbox_from_points(operation.point(point) for point in source_page_corners)

    for primitive in page_data.primitives:
        _transform_primitive(primitive, operation, curve_segments)
    for text in page_data.text_items:
        _transform_text(text, operation)

    page_data.width = page_bounds[2] - page_bounds[0]
    page_data.height = page_bounds[3] - page_bounds[1]
    _disable_source_overlays(page)
    return extraction
