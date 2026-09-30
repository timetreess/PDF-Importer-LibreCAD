# PDF2GeoCAD MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the PRD Version 0.1 manual-GCP georeferencing workflow to the existing PDF-to-DXF importer.

**Architecture:** Keep `pdfcadcore` untouched, solve and apply georeferencing in a new `librecad_pdf_importer.georef` package, and reuse the production importer and DXF exporter. Add only the exporter options needed for metre units and real-coordinate extents.

**Tech Stack:** Python 3.12, numpy, PyMuPDF, ezdxf, pyproj, tkinter, pytest.

---

### Task 1: Georeferencing data and solvers

**Files:**
- Create: `librecad_pdf_importer/georef/__init__.py`
- Create: `librecad_pdf_importer/georef/models.py`
- Create: `librecad_pdf_importer/georef/gcp.py`
- Create: `librecad_pdf_importer/georef/helmert.py`
- Create: `librecad_pdf_importer/georef/affine.py`
- Create: `librecad_pdf_importer/georef/residual.py`
- Test: `tests/test_georef_solvers.py`

- [x] Define immutable GCP, CRS, residual, and transform result models.
- [x] Implement JSON GCP load/save with schema validation.
- [x] Implement rank-checked least-squares Helmert and affine solvers.
- [x] Derive Helmert `scale = hypot(a, b)` and
      `rotation_deg = degrees(atan2(b, a))`; expose coefficients as
      `transform_coefficients.{a,b,tx,ty}` and affine coefficients as
      `transform_coefficients.{a,b,c,d,e,f}`.
- [x] Calculate signed X/Y residuals, total error, RMSE, and maximum error.
- [x] Reject insufficient GCP counts and duplicate, collinear, or otherwise
      rank-deficient source layouts with `GeoreferenceError` before extraction.
- [x] Run `pytest tests/test_georef_solvers.py -q`; expect all tests to pass.

### Task 2: CRS and geometry transformation

**Files:**
- Create: `librecad_pdf_importer/georef/crs.py`
- Create: `librecad_pdf_importer/georef/transform_geometry.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Test: `tests/test_georef_geometry.py`

- [x] Pin `pyproj==3.8.0` in both dependency manifests and validate local/EPSG
      output CRS with `CRS.from_user_input()` / `CRS.to_authority()`.
- [x] Reject invalid or non-EPSG CRS input with `GeoreferenceError` before
      extraction or export.
- [x] Transform primitive points, centres, radii, angles, bounds, text placement,
      positioned characters, line widths, and areas.
- [x] Convert affine-distorted CIRCLE/ARC/RECT entities to polylines.
- [x] Disable source-coordinate-bound overlays and images in georeferenced mode.
- [x] Run `pytest tests/test_georef_geometry.py -q`; expect all tests to pass.

### Task 3: DXF metre mode and georeferenced extents

**Files:**
- Modify: `librecad_pdf_importer/exporters/dxf_exporter.py`
- Test: `tests/test_georef_exporter.py`

- [x] Add `output_units` and `seed_page_extents` to `DxfExportOptions` with
      backward-compatible defaults.
- [x] At exporter lines 4679-4681, select `doc.units`, `$INSUNITS`, and
      `set_raster_variables(..., units=...)` from `output_units` (`mm` or `m`).
- [x] At exporter lines 4782-4785, skip the synthetic `(0, 0)` and page-size
      extent seed when `seed_page_extents=False`, so real-coordinate geometry
      determines `$EXTMIN`/`$EXTMAX`.
- [x] Run `pytest tests/test_georef_exporter.py -q`; expect existing millimetre defaults and new
      metre mode both to pass.

### Task 4: Conversion pipeline and reports

**Files:**
- Create: `librecad_pdf_importer/georef/report.py`
- Create: `librecad_pdf_importer/georef/pipeline.py`
- Test: `tests/test_georef_pipeline.py`

- [x] Extract page 1 through `run_import(..., mode="vector")`.
- [x] Reject extraction results with no supported primitives or text before
      creating any output file.
- [x] Solve and apply the selected transform, then call the production exporter.
- [x] Reopen the exported DXF in `pipeline._attach_dxf_metadata()`, register the
      `PDF2GEOCAD` AppID, add compact status/method/CRS/report XDATA to every
      modelspace entity, and save atomically.
- [x] Emit deterministic JSON and human-readable HTML QC reports. Both formats
      must include every field in design §Outputs and Traceability: source, page,
      timestamp, status, CRS, coefficients, scale, rotation, every GCP and its
      residual, RMSE, maximum residual, threshold result, and output paths.
- [x] Reopen the DXF in the test and verify world coordinates, metre units,
      metadata, and report consistency.
- [x] Run `pytest tests/test_georef_pipeline.py -q`; expect all tests to pass.

### Task 5: CLI surface

**Files:**
- Create: `librecad_pdf_importer/georef/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_georef_cli.py`

- [x] Register `pdf2geocad = "librecad_pdf_importer.georef.cli:main"` and add
      options for GCP file, transform,
      CRS, output directory, threshold, DXF version, and GUI launch.
- [x] Print transform/QC/output summaries and return stable error codes.
- [x] Run CLI help, success, and invalid-input tests.

### Task 6: Tkinter manual-GCP UI

**Files:**
- Create: `librecad_pdf_importer/georef/gui.py`
- Modify: `pyproject.toml`
- Test: `tests/test_georef_gui.py`

- [x] Render the first PDF page on a scrollable canvas.
- [x] Convert click locations to upstream model coordinates and collect world X/Y.
- [x] List, remove, load, and save GCPs.
- [x] Show live scale, rotation, RMSE, maximum residual, and threshold state.
- [x] Export through the same public pipeline used by the CLI and keep Export
      disabled until the selected solver has enough valid GCPs.
- [x] Register
      `pdf2geocad-gui = "librecad_pdf_importer.georef.gui:main"` under
      `[project.gui-scripts]`; `pdf2geocad --gui` remains an equivalent launcher.
- [x] Test click-to-model conversion with both Y directions, live QC updates,
      the insufficient-GCP export gate, and GCP add/remove/load/save round trips.
- [x] Run `pytest tests/test_georef_gui.py -q`; expect all tests to pass without a
      persistent desktop display.

### Task 7: Documentation and release verification

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`

- [x] Document Version 0.1 usage, GCP JSON shape, outputs, and explicit exclusions.
- [x] Run diagnostics on every changed Python file.
- [x] Run all new georef tests, then the full existing test suite.
- [x] Build the wheel with `python -m build` or the repository's supported build
      command and verify importability on Python 3.12.
- [x] Perform end-to-end CLI QA on a generated vector PDF and inspect the DXF,
      JSON, and HTML outputs.
