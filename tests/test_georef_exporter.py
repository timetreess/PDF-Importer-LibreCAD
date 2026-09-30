"""DXF exporter options required by the georeferenced pipeline."""
from __future__ import annotations

import ezdxf
import pytest
from ezdxf.units import M, MM

from librecad_pdf_importer.core.document import DocumentExtraction, ExtractedPage
from librecad_pdf_importer.exporters.dxf_exporter import DxfExportOptions, export_to_dxf
from pdfcadcore.primitives import PageData, Primitive


def _line_extraction(source_path) -> DocumentExtraction:
    page_data = PageData(
        page_number=1,
        width=200.0,
        height=100.0,
        primitives=[
            Primitive(
                id=1,
                type="line",
                points=[(1000.0, 2000.0), (1010.0, 2020.0)],
                page_number=1,
            )
        ],
    )
    return DocumentExtraction(
        pdf_path=str(source_path),
        pages=[ExtractedPage(page_data=page_data, profile=None, resolved_mode="vector")],
        requested_mode="vector",
    )


def _raster_units(doc) -> int:
    raster_variables = list(doc.objects.query("RASTERVARIABLES"))
    assert len(raster_variables) == 1
    return int(raster_variables[0].dxf.units)


def test_export_options_default_to_legacy_millimetres_and_page_seed(tmp_path):
    source = tmp_path / "source.pdf"
    output = tmp_path / "default.dxf"
    options = DxfExportOptions(include_text=False, include_images=False)

    assert options.output_units == "mm"
    assert options.seed_page_extents is True

    export_to_dxf(_line_extraction(source), str(output), options)
    doc = ezdxf.readfile(output)

    assert doc.units == MM
    assert doc.header["$INSUNITS"] == 4
    assert _raster_units(doc) == 1
    assert tuple(doc.header["$EXTMIN"][:2]) == pytest.approx((0.0, 0.0))


def test_export_options_preserve_all_legacy_positional_arguments():
    provenance = object()
    options = DxfExportOptions(
        False,
        "glyphs",
        False,
        False,
        False,
        False,
        "R2013",
        False,
        "overlay",
        0.125,
        provenance,
        "C:/LibreCAD/librecad.exe",
        False,
    )

    assert options.include_text is False
    assert options.text_mode == "glyphs"
    assert options.include_images is False
    assert options.group_by_page is False
    assert options.prefer_source_layers is False
    assert options.attach_metadata is False
    assert options.dxf_version == "R2013"
    assert options.map_dashes is False
    assert options.page_arrangement == "overlay"
    assert options.page_gap_ratio == 0.125
    assert options.provenance_opts is provenance
    assert options.librecad_executable == "C:/LibreCAD/librecad.exe"
    assert options.searchable_text is False
    assert options.output_units == "mm"
    assert options.seed_page_extents is True


def test_metre_mode_uses_geometry_only_for_real_coordinate_extents(tmp_path):
    source = tmp_path / "source.pdf"
    output = tmp_path / "georef.dxf"
    options = DxfExportOptions(
        include_text=False,
        include_images=False,
        output_units="m",
        seed_page_extents=False,
    )

    export_to_dxf(_line_extraction(source), str(output), options)
    doc = ezdxf.readfile(output)

    assert doc.units == M
    assert doc.header["$INSUNITS"] == 6
    assert _raster_units(doc) == 3
    assert tuple(doc.header["$EXTMIN"][:2]) == pytest.approx((1000.0, 2000.0))
    assert tuple(doc.header["$EXTMAX"][:2]) == pytest.approx((1010.0, 2020.0))


def test_export_rejects_unknown_output_units_before_writing(tmp_path):
    output = tmp_path / "invalid.dxf"

    with pytest.raises(ValueError, match="output_units"):
        export_to_dxf(
            _line_extraction(tmp_path / "source.pdf"),
            str(output),
            DxfExportOptions(output_units="feet"),
        )

    assert not output.exists()
