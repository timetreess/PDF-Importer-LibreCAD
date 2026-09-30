from __future__ import annotations

import math

import ezdxf
import pytest

from librecad_pdf_importer.core.document import (
    DocumentExtraction,
    ExtractedPage,
    ImagePlacement,
)
from librecad_pdf_importer.georef import (
    AffineCoefficients,
    GCP,
    GeoreferenceError,
    solve_affine,
    solve_helmert,
    transform_extraction,
)
from librecad_pdf_importer.exporters.dxf_exporter import DxfExportOptions, export_to_dxf
from pdfcadcore.primitives import NormalizedText, PageData, Primitive, TextCharLayout


def _assert_points(actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_point, expected_point in zip(actual, expected, strict=True):
        assert actual_point == pytest.approx(expected_point)


def _helmert_result():
    return solve_helmert(
        (
            GCP(0, 0, 100, 200),
            GCP(10, 0, 100, 220),
            GCP(0, 10, 80, 200),
        )
    )


def _affine_result():
    def world(x, y):
        return 2 * x + y + 10, 0.5 * x + 3 * y - 4

    return solve_affine(
        tuple(GCP(x, y, *world(x, y)) for x, y in ((0, 0), (4, 0), (0, 3), (2, 5)))
    )


def _text_item() -> NormalizedText:
    source_quad = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    character = TextCharLayout(
        text="A",
        glyph_id=65,
        source_origin_pdf=(0.5, 0.2),
        source_bbox_pdf=(0.0, 0.0, 1.0, 1.0),
        source_quad_pdf=source_quad,
        target_origin=(0.5, 0.2),
        target_quad=source_quad,
        advance_width=1.0,
        glyph_height=1.0,
        source_font_size_pdf=10.0,
        source_font_ascender=0.8,
        source_font_descender=-0.2,
        source_writing_mode=0,
    )
    return NormalizedText(
        id=11,
        text="A",
        normalized="A",
        insertion=(1.0, 1.0),
        bbox=(0.0, 0.0, 2.0, 1.0),
        font_size=2.0,
        rotation=0.0,
        source_bbox_pdf=(0.0, 0.0, 2.0, 1.0),
        source_quad_pdf=((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        target_quad_model=((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        advance_width=3.0,
        glyph_height=1.0,
        baseline_descent=0.25,
        source_char_layout=(character,),
        requires_individual_positioning=True,
    )


def _extraction(primitives, *, text_items=()) -> DocumentExtraction:
    page_data = PageData(
        page_number=1,
        width=10.0,
        height=20.0,
        primitives=list(primitives),
        text_items=list(text_items),
        layers=["0"],
        xobject_names=["Im0"],
    )
    page = ExtractedPage(
        page_data=page_data,
        profile=object(),
        images=[ImagePlacement(1, 0, 0, 2, 3, "image.png", 7)],
        image_paint_order=object(),
        source_line_dashes={1: [2.0, 1.0]},
        final_rect_paints=[object()],
        source_capsules=[object()],
        nontext_composites=[object()],
        capsule_paint_order=object(),
        capsule_vector_paint_order=object(),
        display_to_model=(1, 0, 0, 1, 0, 0),
    )
    return DocumentExtraction(pdf_path="drawing.pdf", pages=[page], requested_mode="vector")


def test_source_polyline_transforms_points_bbox_and_closed_state() -> None:
    polyline = Primitive(
        id=1,
        type="polyline",
        points=[(1.0, 1.0), (2.0, 1.0), (3.0, 2.0)],
        bbox=(0.0, 0.0, 4.0, 3.0),
        closed=True,
    )
    extraction = _extraction((polyline,))

    transform_extraction(extraction, _helmert_result())

    assert polyline.type == "polyline"
    _assert_points(polyline.points, [(98.0, 202.0), (98.0, 204.0), (96.0, 206.0)])
    assert polyline.bbox == pytest.approx((94.0, 200.0, 100.0, 208.0))
    assert polyline.closed is True


def test_helmert_transforms_complete_page_geometry_and_preserves_curves() -> None:
    line = Primitive(
        id=1,
        type="line",
        points=[(1.0, 2.0), (3.0, 4.0)],
        bbox=(0.0, 1.0, 4.0, 5.0),
        line_width=0.5,
        area=3.0,
    )
    circle = Primitive(
        id=2,
        type="circle",
        points=[],
        center=(5.0, 5.0),
        radius=2.0,
        bbox=(3.0, 3.0, 7.0, 7.0),
        line_width=0.25,
        area=math.pi * 4.0,
    )
    arc = Primitive(
        id=3,
        type="arc",
        points=[],
        center=(10.0, 10.0),
        radius=5.0,
        start_angle=0.0,
        end_angle=90.0,
        bbox=(5.0, 5.0, 15.0, 15.0),
    )
    text = _text_item()
    extraction = _extraction((line, circle, arc), text_items=(text,))

    returned = transform_extraction(extraction, _helmert_result())

    assert returned is extraction
    _assert_points(line.points, [(96.0, 202.0), (92.0, 206.0)])
    assert line.bbox == pytest.approx((90.0, 200.0, 98.0, 208.0))
    assert line.line_width == pytest.approx(1.0)
    assert line.area == pytest.approx(12.0)

    assert circle.type == "circle"
    assert circle.center == pytest.approx((90.0, 210.0))
    assert circle.radius == pytest.approx(4.0)
    assert circle.bbox == pytest.approx((86.0, 206.0, 94.0, 214.0))
    assert circle.area == pytest.approx(math.pi * 16.0)

    assert arc.type == "arc"
    assert arc.center == pytest.approx((80.0, 220.0))
    assert arc.radius == pytest.approx(10.0)
    assert arc.start_angle == pytest.approx(90.0)
    assert arc.end_angle == pytest.approx(180.0)

    assert text.insertion == pytest.approx((98.0, 202.0))
    assert text.bbox == pytest.approx((98.0, 200.0, 100.0, 204.0))
    assert text.rotation == pytest.approx(90.0)
    assert text.font_size == pytest.approx(4.0)
    assert text.advance_width == pytest.approx(6.0)
    assert text.glyph_height == pytest.approx(2.0)
    assert text.baseline_descent == pytest.approx(0.5)
    _assert_points(
        text.target_quad_model,
        ((100.0, 200.0), (100.0, 204.0), (98.0, 204.0), (98.0, 200.0))
    )
    assert text.source_bbox_pdf == (0.0, 0.0, 2.0, 1.0)
    assert text.source_quad_pdf == (
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (0.0, 1.0),
    )
    character = text.source_char_layout[0]
    assert character.target_origin == pytest.approx((99.6, 201.0))
    _assert_points(
        character.target_quad,
        ((100.0, 200.0), (100.0, 202.0), (98.0, 202.0), (98.0, 200.0))
    )
    assert character.advance_width == pytest.approx(2.0)
    assert character.glyph_height == pytest.approx(2.0)
    assert character.source_quad_pdf == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )

    page = extraction.pages[0]
    assert (page.page_data.width, page.page_data.height) == pytest.approx((40.0, 20.0))
    assert page.page_data.xobject_names == []
    assert page.images == []
    assert page.image_paint_order is None
    assert page.source_line_dashes == {}
    assert page.final_rect_paints == []
    assert page.source_capsules == []
    assert page.nontext_composites == []
    assert page.capsule_paint_order is None
    assert page.capsule_vector_paint_order is None
    assert page.display_to_model is None


def test_transformed_text_preserves_source_provenance_and_exports(tmp_path) -> None:
    text = _text_item()
    source_bbox = text.source_bbox_pdf
    source_quad = text.source_quad_pdf
    extraction = _extraction((), text_items=(text,))
    output = tmp_path / "world-text.dxf"

    transform_extraction(extraction, _helmert_result())
    result = export_to_dxf(
        extraction,
        str(output),
        DxfExportOptions(
            text_mode="labels",
            include_images=False,
            attach_metadata=False,
            searchable_text=False,
        ),
    )

    assert text.source_bbox_pdf == source_bbox
    assert text.source_quad_pdf == source_quad
    assert result.text_deliveries[0]["dropped"] is False
    drawing = ezdxf.readfile(output)
    assert not drawing.audit().errors
    text_entities = list(drawing.modelspace().query("TEXT"))
    assert [entity.plain_text() for entity in text_entities] == ["A"]


def test_general_affine_degrades_curves_and_rotated_rects_to_polylines() -> None:
    circle = Primitive(
        id=1,
        type="circle",
        points=[],
        center=(1.0, 2.0),
        radius=2.0,
        bbox=(-1.0, 0.0, 3.0, 4.0),
        line_width=0.4,
        area=math.pi * 4.0,
    )
    arc = Primitive(
        id=2,
        type="arc",
        points=[],
        center=(0.0, 0.0),
        radius=1.0,
        start_angle=0.0,
        end_angle=90.0,
    )
    rect = Primitive(
        id=3,
        type="rect",
        points=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
        bbox=(0.0, 0.0, 2.0, 1.0),
        area=2.0,
    )
    text = _text_item()
    extraction = _extraction((circle, arc, rect), text_items=(text,))

    transform_extraction(extraction, _affine_result(), curve_segments=16)

    assert circle.type == "polyline"
    assert circle.closed is True
    assert len(circle.points) == 16
    assert circle.points[0] == pytest.approx((18.0, 3.5))
    assert circle.center is None
    assert circle.radius is None
    assert circle.start_angle is None
    assert circle.end_angle is None
    assert circle.area == pytest.approx(math.pi * 4.0 * 5.5)
    assert circle.line_width == pytest.approx(0.4 * math.sqrt(5.5))
    assert circle.bbox == pytest.approx(
        (
            min(point[0] for point in circle.points),
            min(point[1] for point in circle.points),
            max(point[0] for point in circle.points),
            max(point[1] for point in circle.points),
        )
    )

    assert arc.type == "polyline"
    assert arc.closed is False
    assert len(arc.points) == 17
    assert arc.points[0] == pytest.approx((12.0, -3.5))
    assert arc.points[-1] == pytest.approx((11.0, -1.0))
    assert arc.center is None
    assert arc.radius is None
    assert arc.start_angle is None
    assert arc.end_angle is None

    assert rect.type == "polyline"
    assert rect.closed is True
    _assert_points(
        rect.points,
        [(10.0, -4.0), (14.0, -3.0), (15.0, 0.0), (11.0, -1.0)]
    )
    assert rect.area == pytest.approx(11.0)

    baseline_scale = math.hypot(2.0, 0.5)
    vertical_scale = math.hypot(1.0, 3.0)
    nominal_scale = math.sqrt(5.5)
    assert text.insertion == pytest.approx((13.0, -0.5))
    assert text.rotation == pytest.approx(math.degrees(math.atan2(0.5, 2.0)))
    assert text.font_size == pytest.approx(2.0 * nominal_scale)
    assert text.advance_width == pytest.approx(3.0 * baseline_scale)
    assert text.glyph_height == pytest.approx(vertical_scale)
    assert text.baseline_descent == pytest.approx(0.25 * vertical_scale)
    _assert_points(
        text.target_quad_model,
        ((10.0, -4.0), (14.0, -3.0), (15.0, 0.0), (11.0, -1.0))
    )


def test_similarity_affine_preserves_circles_and_arcs() -> None:
    def world(x, y):
        return -3 * y + 5, 3 * x + 7

    result = solve_affine(
        tuple(GCP(x, y, *world(x, y)) for x, y in ((0, 0), (1, 0), (0, 1)))
    )
    circle = Primitive(id=1, type="circle", points=[], center=(2, 1), radius=4)
    arc = Primitive(
        id=2,
        type="arc",
        points=[],
        center=(0, 0),
        radius=2,
        start_angle=0,
        end_angle=180,
    )
    extraction = _extraction((circle, arc))

    transform_extraction(extraction, result)

    assert circle.type == "circle"
    assert circle.center == pytest.approx((2.0, 13.0))
    assert circle.radius == pytest.approx(12.0)
    assert arc.type == "arc"
    assert arc.start_angle == pytest.approx(90.0)
    assert arc.end_angle == pytest.approx(270.0)


def test_tiny_anisotropic_affine_degrades_circle_to_polyline() -> None:
    circle = Primitive(id=1, type="circle", points=[], center=(0, 0), radius=1)
    extraction = _extraction((circle,))

    transform_extraction(
        extraction,
        AffineCoefficients(1e-9, 0.0, 0.0, 0.0, 2e-9, 0.0),
        curve_segments=8,
    )

    assert circle.type == "polyline"
    assert circle.closed is True
    assert len(circle.points) == 8


@pytest.mark.parametrize(
    "coefficients",
    [
        pytest.param(
            AffineCoefficients(1e-18, 0.0, 0.0, 0.0, 1e-18, 0.0),
            id="tiny",
        ),
        pytest.param(
            AffineCoefficients(-2.0, 0.0, 5.0, 0.0, 2.0, 7.0),
            id="reflected",
        ),
    ],
)
def test_tiny_and_reflected_similarities_preserve_curves(coefficients) -> None:
    circle = Primitive(id=1, type="circle", points=[], center=(2, 1), radius=4)
    arc = Primitive(
        id=2,
        type="arc",
        points=[],
        center=(0, 0),
        radius=2,
        start_angle=0,
        end_angle=90,
    )
    extraction = _extraction((circle, arc))

    transform_extraction(extraction, coefficients)

    assert circle.type == "circle"
    assert circle.radius is not None
    assert arc.type == "arc"
    assert arc.start_angle is not None
    assert arc.end_angle is not None


@pytest.mark.parametrize("page_count", [0, 2])
def test_transform_extraction_requires_exactly_one_page(page_count) -> None:
    extraction = DocumentExtraction(
        pdf_path="drawing.pdf",
        pages=[_extraction(()).pages[0] for _ in range(page_count)],
    )
    with pytest.raises(GeoreferenceError, match="exactly one"):
        transform_extraction(extraction, _helmert_result())
