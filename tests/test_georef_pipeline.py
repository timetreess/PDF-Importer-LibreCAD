"""End-to-end georeferenced conversion pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import pymupdf as fitz
import pytest

from librecad_pdf_importer.georef.gcp import save_gcps
from librecad_pdf_importer.georef.models import GCP, GeoreferenceError
from librecad_pdf_importer.georef import pipeline
from librecad_pdf_importer.georef.pipeline import APP_ID, run_georef_pipeline


MM_PER_PT = 25.4 / 72.0


def _write_vector_pdf(path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.draw_line((20, 30), (100, 80), color=(0, 0, 0), width=1)
    document.save(path)
    document.close()
    return (
        (20 * MM_PER_PT, (100 - 30) * MM_PER_PT),
        (100 * MM_PER_PT, (100 - 80) * MM_PER_PT),
    )


def _world(point: tuple[float, float]) -> tuple[float, float]:
    return 2.0 * point[0] + 1000.0, 2.0 * point[1] + 2000.0


def _write_helmert_gcps(path: Path, source_points) -> tuple[GCP, ...]:
    gcps = tuple(
        GCP(
            source_x=source_x,
            source_y=source_y,
            world_x=_world((source_x, source_y))[0],
            world_y=_world((source_x, source_y))[1],
            label=f"P{index}",
        )
        for index, (source_x, source_y) in enumerate(source_points, start=1)
    )
    save_gcps(path, gcps)
    return gcps


def test_pipeline_writes_world_dxf_xdata_and_consistent_reports(tmp_path):
    source = tmp_path / "drawing.pdf"
    source_points = _write_vector_pdf(source)
    gcp_path = tmp_path / "drawing.gcps.json"
    gcps = _write_helmert_gcps(gcp_path, source_points)
    output_dir = tmp_path / "out"

    result = run_georef_pipeline(
        source,
        gcp_path,
        transform="helmert",
        crs="local",
        output_dir=output_dir,
        rmse_threshold=0.01,
        include_text=False,
    )

    assert result.dxf_path == output_dir / "drawing_georef.dxf"
    assert result.json_path == output_dir / "drawing_georef.json"
    assert result.html_path == output_dir / "drawing_georef_report.html"
    assert all(path.is_file() for path in result.output_paths)

    drawing = ezdxf.readfile(result.dxf_path)
    assert drawing.units == 6
    assert drawing.header["$INSUNITS"] == 6
    assert APP_ID in drawing.appids
    entities = list(drawing.modelspace())
    assert entities
    line = next(entity for entity in entities if entity.dxftype() == "LINE")
    assert (line.dxf.start.x, line.dxf.start.y) == pytest.approx(
        _world(source_points[0]), abs=1e-7
    )
    assert (line.dxf.end.x, line.dxf.end.y) == pytest.approx(
        _world(source_points[1]), abs=1e-7
    )
    extmin = drawing.header["$EXTMIN"]
    assert (extmin[0], extmin[1]) != pytest.approx((0.0, 0.0))
    for entity in entities:
        values = [tag.value for tag in entity.get_xdata(APP_ID)]
        assert values == [
            "status=Calibrated",
            "method=helmert",
            "crs=local",
            "report=drawing_georef.json",
        ]

    report = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert report == result.report
    assert report["source"] == str(source.resolve())
    assert report["page"] == 1
    assert report["timestamp_utc"].endswith("Z")
    assert report["status"] == "Calibrated"
    assert report["crs"]["label"] == "local"
    assert report["method"] == "helmert"
    assert report["coefficients"] == pytest.approx(
        {"a": 2.0, "b": 0.0, "tx": 1000.0, "ty": 2000.0}, abs=1e-9
    )
    assert report["scale"] == pytest.approx(2.0)
    assert report["rotation_deg"] == pytest.approx(0.0, abs=1e-9)
    assert len(report["gcps"]) == len(gcps)
    assert report["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert report["max_residual"] == pytest.approx(0.0, abs=1e-9)
    assert report["rmse_threshold"] == 0.01
    assert report["threshold_status"] == "pass"
    assert report["outputs"] == {
        "dxf": str(result.dxf_path.resolve()),
        "json": str(result.json_path.resolve()),
        "html": str(result.html_path.resolve()),
    }
    assert all(row["residual"]["total"] == pytest.approx(0.0, abs=1e-9) for row in report["gcps"])

    html = result.html_path.read_text(encoding="utf-8")
    for expected in (
        str(source.resolve()),
        report["timestamp_utc"],
        "Calibrated",
        "helmert",
        "Local coordinates",
        "RMSE",
        "Maximum residual",
        "pass",
        "drawing_georef.dxf",
        "drawing_georef.json",
    ):
        assert expected in html


@pytest.mark.parametrize(
    ("gcp_count", "crs", "message"),
    [
        (1, "local", "at least 2 GCPs"),
        (2, "EPSG:4326", "projected 2D CRS"),
    ],
)
def test_pipeline_validates_gcps_and_crs_before_extraction(
    tmp_path, monkeypatch, gcp_count, crs, message
):
    source = tmp_path / "drawing.pdf"
    source_points = _write_vector_pdf(source)
    gcp_path = tmp_path / "drawing.gcps.json"
    _write_helmert_gcps(gcp_path, source_points[:gcp_count])
    output_dir = tmp_path / "out"

    def unexpected_import(*_args, **_kwargs):
        raise AssertionError("extraction must not start")

    monkeypatch.setattr(pipeline, "run_import", unexpected_import)

    with pytest.raises(GeoreferenceError, match=message):
        run_georef_pipeline(source, gcp_path, crs=crs, output_dir=output_dir)

    assert not output_dir.exists()


def test_pipeline_rejects_r12_before_extraction_or_artifact_creation(tmp_path, monkeypatch):
    source = tmp_path / "drawing.pdf"
    source_points = _write_vector_pdf(source)
    gcp_path = tmp_path / "drawing.gcps.json"
    _write_helmert_gcps(gcp_path, source_points)
    output_dir = tmp_path / "out"

    def unexpected_import(*_args, **_kwargs):
        raise AssertionError("extraction must not start")

    monkeypatch.setattr(pipeline, "run_import", unexpected_import)

    with pytest.raises(GeoreferenceError, match="R2000 or newer"):
        run_georef_pipeline(
            source,
            gcp_path,
            output_dir=output_dir,
            dxf_version="R12",
        )

    assert not output_dir.exists()


def test_pipeline_extracts_only_page_one_without_images_or_raster_fallback(
    tmp_path, monkeypatch
):
    source = tmp_path / "blank.pdf"
    document = fitz.open()
    document.new_page(width=100, height=100)
    document.new_page(width=100, height=100)
    document.save(source)
    document.close()
    gcp_path = tmp_path / "blank.gcps.json"
    _write_helmert_gcps(gcp_path, ((0.0, 0.0), (1.0, 0.0)))
    output_dir = tmp_path / "out"
    actual_run_import = pipeline.run_import
    captured = {}

    def recording_import(pdf_path, mode="auto", overrides=None):
        captured.update(pdf_path=pdf_path, mode=mode, overrides=dict(overrides or {}))
        return actual_run_import(pdf_path, mode=mode, overrides=overrides)

    monkeypatch.setattr(pipeline, "run_import", recording_import)

    with pytest.raises(GeoreferenceError, match="no supported vector primitive or text"):
        run_georef_pipeline(
            source,
            gcp_path,
            output_dir=output_dir,
            include_text=False,
        )

    assert captured == {
        "pdf_path": str(source.resolve()),
        "mode": "vector",
        "overrides": {
            "pages": "1",
            "ignore_images": True,
            "raster_fallback": False,
            "import_text": False,
            "text_mode": "none",
        },
    }
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_pipeline_publish_failure_restores_prior_artifacts_and_removes_staging(
    tmp_path, monkeypatch
):
    source = tmp_path / "drawing.pdf"
    source_points = _write_vector_pdf(source)
    gcp_path = tmp_path / "drawing.gcps.json"
    _write_helmert_gcps(gcp_path, source_points)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    prior = {
        output_dir / "drawing_georef.dxf": b"prior dxf",
        output_dir / "drawing_georef.json": b"prior json",
        output_dir / "drawing_georef_report.html": b"prior html",
    }
    for path, content in prior.items():
        path.write_bytes(content)

    actual_replace = pipeline.os.replace
    failed = False

    def fail_json_publish(source_path, destination_path):
        nonlocal failed
        destination = Path(destination_path)
        if not failed and destination == output_dir / "drawing_georef.json":
            failed = True
            raise OSError("injected JSON publish failure")
        return actual_replace(source_path, destination_path)

    monkeypatch.setattr(pipeline.os, "replace", fail_json_publish)

    with pytest.raises(OSError, match="injected JSON publish failure"):
        run_georef_pipeline(
            source,
            gcp_path,
            output_dir=output_dir,
            include_text=False,
        )

    assert failed
    assert {path: path.read_bytes() for path in prior} == prior
    assert set(output_dir.iterdir()) == set(prior)


def test_pipeline_exports_with_warning_when_rmse_exceeds_threshold(tmp_path):
    source = tmp_path / "drawing.pdf"
    source_points = _write_vector_pdf(source)
    gcp_path = tmp_path / "drawing.gcps.json"
    points = (*source_points, (0.0, 0.0))
    gcps = []
    for index, point in enumerate(points, start=1):
        world_x, world_y = _world(point)
        if index == 3:
            world_x += 1.0
        gcps.append(GCP(point[0], point[1], world_x, world_y, f"P{index}"))
    save_gcps(gcp_path, gcps)

    result = run_georef_pipeline(
        source,
        gcp_path,
        output_dir=tmp_path / "out",
        rmse_threshold=0.01,
        include_text=False,
    )

    assert result.report["rmse"] > result.report["rmse_threshold"]
    assert result.report["threshold_status"] == "warning"
    assert all(path.is_file() for path in result.output_paths)
    assert "warning" in result.html_path.read_text(encoding="utf-8")
