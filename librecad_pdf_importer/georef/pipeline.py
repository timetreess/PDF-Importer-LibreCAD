"""End-to-end manual-GCP PDF georeferencing pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import uuid

import ezdxf

from librecad_pdf_importer.exporters.dxf_exporter import DxfExportOptions, export_to_dxf
from librecad_pdf_importer.importer import run_import

from .affine import solve_affine
from .crs import validate_crs
from .gcp import load_gcps
from .helmert import solve_helmert
from .models import CRSDefinition, GeoreferenceError, TransformResult
from .report import build_report, write_html_report, write_json_report
from .transform_geometry import transform_extraction


APP_ID = "PDF2GEOCAD"
GEOREF_DXF_VERSIONS = ("R2000", "R2004", "R2007", "R2010", "R2013", "R2018")
_SUPPORTED_PRIMITIVES = {"line", "polyline", "closed_loop", "arc", "circle", "rect"}


@dataclass(frozen=True)
class GeorefPipelineResult:
    dxf_path: Path
    json_path: Path
    html_path: Path
    transform: TransformResult
    crs: CRSDefinition
    report: dict[str, Any]

    @property
    def output_paths(self) -> tuple[Path, Path, Path]:
        return self.dxf_path, self.json_path, self.html_path


def _solve(transform: str, gcps) -> TransformResult:
    method = str(transform).strip().lower()
    if method == "helmert":
        return solve_helmert(gcps)
    if method == "affine":
        return solve_affine(gcps)
    raise GeoreferenceError("Transform must be 'helmert' or 'affine'")


def _crs_label(crs: CRSDefinition) -> str:
    return crs.authority or "local"


def _attach_dxf_metadata(
    path: Path,
    *,
    status: str,
    method: str,
    crs_label: str,
    report_filename: str,
) -> None:
    """Reopen, annotate, verify, and atomically replace a staged DXF."""

    drawing = ezdxf.readfile(path)
    if APP_ID not in drawing.appids:
        drawing.appids.add(APP_ID)
    xdata = [
        (1000, f"status={status}"),
        (1000, f"method={method}"),
        (1000, f"crs={crs_label}"),
        (1000, f"report={report_filename}"),
    ]
    for entity in drawing.modelspace():
        entity.set_xdata(APP_ID, xdata)

    temporary = path.with_name(f".{path.stem[:12]}.{uuid.uuid4().hex[:12]}.tmp.dxf")
    try:
        drawing.saveas(temporary)
        verified = ezdxf.readfile(temporary)
        if APP_ID not in verified.appids:
            raise GeoreferenceError("DXF metadata verification failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _has_supported_content(extraction, *, include_text: bool) -> bool:
    for page in extraction.pages:
        if any(
            str(primitive.type).strip().lower() in _SUPPORTED_PRIMITIVES
            for primitive in page.page_data.primitives
        ):
            return True
        if include_text and any(str(item.text) for item in page.page_data.text_items):
            return True
    return False


def _publish_artifacts(pairs: tuple[tuple[Path, Path], ...], staging_dir: Path) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, (_staged, destination) in enumerate(pairs):
            if destination.exists():
                backup = staging_dir / f"backup-{index}-{destination.name}"
                os.replace(destination, backup)
                backups.append((destination, backup))
        for staged, destination in pairs:
            os.replace(staged, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, destination)
        raise


def run_georef_pipeline(
    pdf_path: str | Path,
    gcp_path: str | Path,
    *,
    transform: str = "helmert",
    crs: str | int | None = "local",
    output_dir: str | Path | None = None,
    rmse_threshold: float = 1.0,
    dxf_version: str = "R2018",
    include_text: bool = True,
    text_mode: str = "text",
) -> GeorefPipelineResult:
    """Convert page 1 of one vector PDF into calibrated metre-based artifacts."""

    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file():
        raise GeoreferenceError(f"Input PDF not found: {source}")
    normalized_dxf_version = str(dxf_version).strip().upper()
    if normalized_dxf_version not in GEOREF_DXF_VERSIONS:
        raise GeoreferenceError("Georeferenced DXF version must be R2000 or newer")
    try:
        threshold = float(rmse_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeoreferenceError("RMSE threshold must be a finite non-negative number") from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise GeoreferenceError("RMSE threshold must be a finite non-negative number")

    gcps = load_gcps(gcp_path)
    solved = _solve(transform, gcps)
    crs_definition = validate_crs(crs)

    destination_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else source.parent
    )
    stem = source.stem
    dxf_path = destination_dir / f"{stem}_georef.dxf"
    json_path = destination_dir / f"{stem}_georef.json"
    html_path = destination_dir / f"{stem}_georef_report.html"
    destination_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f".{stem[:12]}-georef-", dir=destination_dir) as temporary:
        staging_dir = Path(temporary)
        staged_dxf = staging_dir / dxf_path.name
        staged_json = staging_dir / json_path.name
        staged_html = staging_dir / html_path.name
        overrides = {
            "pages": "1",
            "ignore_images": True,
            "raster_fallback": False,
            "import_text": bool(include_text),
            "text_mode": text_mode if include_text else "none",
        }
        with run_import(str(source), mode="vector", overrides=overrides) as run:
            extraction = run.extraction
            if not _has_supported_content(extraction, include_text=bool(include_text)):
                raise GeoreferenceError(
                    "Page 1 contains no supported vector primitive or text"
                )
            transform_extraction(extraction, solved)
            export_to_dxf(
                extraction,
                str(staged_dxf),
                DxfExportOptions(
                    include_text=bool(include_text),
                    text_mode=text_mode if include_text else "none",
                    include_images=False,
                    group_by_page=False,
                    dxf_version=normalized_dxf_version,
                    searchable_text=False,
                    output_units="m",
                    seed_page_extents=False,
                ),
            )

        _attach_dxf_metadata(
            staged_dxf,
            status="Calibrated",
            method=solved.method,
            crs_label=_crs_label(crs_definition),
            report_filename=json_path.name,
        )
        outputs = {"dxf": dxf_path, "json": json_path, "html": html_path}
        report = build_report(
            source=source,
            transform=solved,
            crs=crs_definition,
            rmse_threshold=threshold,
            outputs=outputs,
        )
        write_json_report(staged_json, report)
        write_html_report(staged_html, report)
        _publish_artifacts(
            (
                (staged_dxf, dxf_path),
                (staged_json, json_path),
                (staged_html, html_path),
            ),
            staging_dir,
        )

    return GeorefPipelineResult(
        dxf_path=dxf_path,
        json_path=json_path,
        html_path=html_path,
        transform=solved,
        crs=crs_definition,
        report=report,
    )
