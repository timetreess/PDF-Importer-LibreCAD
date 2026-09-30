# PDF2GeoCAD MVP Design

## Goal

Implement PRD Version 0.1 on top of `BlueCollar-Systems/PDF-Importer-LibreCAD` at
commit `34c410b5688cefa8ce86e384e37551b3231ea333`. A user can open one vector-PDF
page, register ground-control points (GCPs), solve a Helmert or affine transform,
inspect residuals, and export a metre-based real-coordinate DXF plus JSON and HTML
traceability reports.

## Scope

Included:

- One vector-PDF page.
- Existing LINE, POLYLINE, ARC, CIRCLE, and TEXT extraction.
- Manual GCP entry from CLI JSON or clicks in a Tkinter PDF viewer.
- Least-squares 2D Helmert for two or more GCPs.
- Least-squares affine transform for three or more GCPs.
- Per-GCP residuals, RMSE, maximum residual, and configurable warning threshold.
- Local coordinates or a validated EPSG CRS label.
- DXF in metres with `Calibrated` metadata.
- `<stem>_georef.dxf`, `<stem>_georef.json`, and
  `<stem>_georef_report.html` outputs.

Excluded by PRD Version 0.1:

- GeoPDF auto-detection, GeoPackage, multiple pages, QGIS loading, map overlay,
  OCR, raster vectorization, automatic coordinate recognition, RANSAC, and DWG.

## Architecture

The shared `pdfcadcore/` directory remains byte-identical to upstream. The new
`librecad_pdf_importer/georef/` package owns GCP persistence, transform solving,
geometry transformation, CRS validation, reporting, and the georeferenced
conversion pipeline.

```text
PDF
  -> existing run_import(..., mode="vector")
  -> DocumentExtraction / PageData in millimetres
  -> georef solver from GCPs
  -> in-place world-coordinate geometry transformation
  -> existing export_to_dxf(..., output_units="m")
  -> DXF metadata post-processing
  -> JSON + HTML QC reports
```

The existing `pdf2dxf` and `lcpdf-gui` behavior remains unchanged. A new
`pdf2geocad` command and `pdf2geocad-gui` entry point expose the MVP.

## Coordinate Contract

- Extracted model coordinates are millimetres, with Y increasing upward because
  the upstream `ImportConfig.flip_y` default is true.
- GUI clicks are captured in rendered page coordinates and converted to the same
  model space before creating a GCP.
- GCP world coordinates are already expressed in the chosen output CRS or local
  system; no implicit reprojection occurs.
- EPSG mode validates the identifier with `pyproj.CRS` and records its canonical
  authority string. Local mode records no EPSG code.
- DXF coordinates are the solved world coordinates and `$INSUNITS` is metres.

## Transform Behavior

Helmert uses:

```text
X = a*x - b*y + tx
Y = b*x + a*y + ty
```

Affine uses:

```text
X = a*x + b*y + c
Y = d*x + e*y + f
```

Rank-deficient GCP layouts are rejected. Residuals are evaluated in output units.
Helmert preserves circles and arcs. Affine preserves them only when its linear
part is a similarity transform; otherwise circles and arcs are sampled into
polylines so no invalid DXF CIRCLE/ARC is emitted. Rotated rectangles are likewise
represented as polylines.

The upstream source-bound visual proof structures depend on pre-transform PDF
coordinates. Georeferenced conversion disables those specialised overlays and
exports the transformed editable primitives instead. Raster images are excluded
from Version 0.1.

## User Interfaces

### CLI

```text
pdf2geocad drawing.pdf --gcp drawing.gcps.json --transform helmert
```

The GCP file contains source model coordinates and world coordinates. CLI options
select output directory, transform, CRS/local mode, RMSE threshold, DXF version,
and text inclusion. `--gui` opens the desktop workflow.

### Tkinter GUI

The GUI opens the first PDF page, renders it on a scrollable canvas, and lets the
user click a point and enter world X/Y. A table lists GCPs and supports deletion,
load, and save. Once enough points exist, the GUI continuously shows method,
scale/rotation for Helmert, RMSE, maximum residual, and threshold status. Export
is disabled until the selected transform is solvable.

## Outputs and Traceability

The JSON report is the machine-readable source of truth. It records source path,
page, timestamp, coordinate status, CRS, transform coefficients, scale, rotation,
all GCPs and residuals, RMSE, maximum residual, threshold status, and output paths.
The HTML report renders the same information for review.

Every DXF modelspace entity receives compact `PDF2GEOCAD` XDATA containing the
coordinate status, method, CRS label, and report filename. This makes the
calibration state discoverable from the CAD artifact without overloading XDATA.

## Failure and Warning Rules

- Fewer than two Helmert or three affine GCPs: stop before extraction/export.
- Duplicate or rank-deficient source GCP geometry: stop with an actionable error.
- Invalid EPSG code: stop before export.
- RMSE above threshold: export succeeds but JSON, HTML, GUI, and CLI show a warning.
- Non-vector or empty page: stop when extraction yields no supported primitives or
  text.

## Verification

- Solver tests recover known Helmert and affine coefficients and reject degenerate
  GCPs.
- Geometry tests cover LINE, polyline, CIRCLE/ARC preservation, affine
  circle-to-polyline conversion, text placement, and bounding boxes.
- Pipeline test creates a synthetic vector PDF, transforms it through the public
  pipeline, reopens the DXF with ezdxf, and verifies metre units and world
  coordinates.
- CLI `--help`, one successful conversion, and one invalid-GCP invocation are run.
- GUI construction and click-coordinate conversion are tested without requiring a
  persistent display; final manual QA drives both the installed CLI and real Tk widgets.
