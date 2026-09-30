"""Machine-readable and human-readable georeferencing QC reports."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

from pdfcadcore.atomic_io import atomic_write_text

from .models import CRSDefinition, TransformResult


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_report(
    *,
    source: Path,
    transform: TransformResult,
    crs: CRSDefinition,
    rmse_threshold: float,
    outputs: Mapping[str, Path],
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build the shared JSON/HTML report payload."""

    threshold_status = "warning" if transform.rmse > rmse_threshold else "pass"
    crs_label = crs.authority or "local"
    residual_rows = []
    for residual in transform.residuals:
        gcp = residual.gcp
        residual_rows.append(
            {
                "index": residual.gcp_index,
                "label": gcp.label,
                "source": {"x": gcp.source_x, "y": gcp.source_y},
                "world": {"x": gcp.world_x, "y": gcp.world_y},
                "residual": {
                    "x": residual.residual_x,
                    "y": residual.residual_y,
                    "total": residual.total_error,
                },
            }
        )

    return {
        "source": str(source.resolve()),
        "page": 1,
        "timestamp_utc": timestamp_utc or _utc_timestamp(),
        "status": "Calibrated",
        "crs": {
            "mode": crs.mode,
            "authority": crs.authority,
            "name": crs.name,
            "label": crs_label,
        },
        "method": transform.method,
        "coefficients": asdict(transform.transform_coefficients),
        "scale": transform.scale,
        "rotation_deg": transform.rotation_deg,
        "gcps": residual_rows,
        "rmse": transform.rmse,
        "max_residual": transform.max_residual,
        "rmse_threshold": rmse_threshold,
        "threshold_status": threshold_status,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }


def write_json_report(path: Path, report: Mapping[str, Any]) -> str:
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return atomic_write_text(path, content, encoding="utf-8")


def render_html_report(report: Mapping[str, Any]) -> str:
    """Render every traceability field from the JSON source of truth."""

    def value(item: Any) -> str:
        if item is None:
            return "—"
        return escape(str(item))

    coefficients = report["coefficients"]
    coefficient_rows = "".join(
        f"<tr><th>{escape(str(name))}</th><td>{value(number)}</td></tr>"
        for name, number in coefficients.items()
    )
    gcp_rows = "".join(
        "<tr>"
        f"<td>{value(row['index'])}</td><td>{value(row['label'])}</td>"
        f"<td>{value(row['source']['x'])}</td><td>{value(row['source']['y'])}</td>"
        f"<td>{value(row['world']['x'])}</td><td>{value(row['world']['y'])}</td>"
        f"<td>{value(row['residual']['x'])}</td><td>{value(row['residual']['y'])}</td>"
        f"<td>{value(row['residual']['total'])}</td>"
        "</tr>"
        for row in report["gcps"]
    )
    output_rows = "".join(
        f"<tr><th>{escape(str(name).upper())}</th><td>{value(path)}</td></tr>"
        for name, path in report["outputs"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDF2GeoCAD QC report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0 0 1.5rem; }}
    th, td {{ border: 1px solid #ccd1d1; padding: .45rem .6rem; text-align: left; }}
    th {{ background: #f4f6f7; }}
    .pass {{ color: #176b35; }} .warning {{ color: #9a5b00; }}
  </style>
</head>
<body>
  <h1>PDF2GeoCAD QC report</h1>
  <table>
    <tr><th>Source</th><td>{value(report['source'])}</td></tr>
    <tr><th>Page</th><td>{value(report['page'])}</td></tr>
    <tr><th>UTC timestamp</th><td>{value(report['timestamp_utc'])}</td></tr>
    <tr><th>Status</th><td>{value(report['status'])}</td></tr>
    <tr><th>CRS</th><td>{value(report['crs']['label'])} — {value(report['crs']['name'])}</td></tr>
    <tr><th>Method</th><td>{value(report['method'])}</td></tr>
    <tr><th>Scale</th><td>{value(report['scale'])}</td></tr>
    <tr><th>Rotation (degrees)</th><td>{value(report['rotation_deg'])}</td></tr>
    <tr><th>RMSE</th><td>{value(report['rmse'])}</td></tr>
    <tr><th>Maximum residual</th><td>{value(report['max_residual'])}</td></tr>
    <tr><th>RMSE threshold</th><td>{value(report['rmse_threshold'])}</td></tr>
    <tr><th>Threshold result</th><td class="{value(report['threshold_status'])}">{value(report['threshold_status'])}</td></tr>
  </table>
  <h2>Transform coefficients</h2>
  <table>{coefficient_rows}</table>
  <h2>Ground control points and residuals</h2>
  <table>
    <thead><tr><th>#</th><th>Label</th><th>Source X</th><th>Source Y</th><th>World X</th><th>World Y</th><th>Residual X</th><th>Residual Y</th><th>Total</th></tr></thead>
    <tbody>{gcp_rows}</tbody>
  </table>
  <h2>Outputs</h2>
  <table>{output_rows}</table>
</body>
</html>
"""


def write_html_report(path: Path, report: Mapping[str, Any]) -> str:
    return atomic_write_text(path, render_html_report(report), encoding="utf-8")
