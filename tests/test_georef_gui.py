"""Headless tests for the PDF2GeoCAD Tkinter GCP-picking GUI.

The GUI is a small industrial Tkinter desktop tool. These tests cover its
pure helper functions and structural Tkinter wiring so they can run on a
continuous-integration agent that does not have a persistent display. The
upstream georeferencing pipeline and DXF exporter are exercised through the
public :func:`librecad_pdf_importer.georef.pipeline.run_georef_pipeline`
helper and are not re-implemented here.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf as fitz
import pytest

pytest.importorskip("tkinter")

import tkinter as tk

from librecad_pdf_importer.georef import gui
from librecad_pdf_importer.georef.gcp import save_gcps
from librecad_pdf_importer.georef.models import GCP


MM_PER_PT = 25.4 / 72.0


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_vector_pdf(path: Path) -> None:
    """Write a deterministic 200x100 pt vector PDF."""
    document = fitz.open()
    document.new_page(width=200, height=100)
    document.save(path)
    document.close()


def _create_app(pdf_path: Path, output_dir: Path):
    """Create a GeorefGuiApp, retrying once if the Tcl interpreter is stale.

    On Windows, rapid create/destroy of ``tk.Tk`` roots can leave the embedded
    Tcl interpreter in a partially initialised state (missing ``tk_library``,
    broken ``auto_path``). Forcing garbage collection between attempts is
    usually enough to recover.
    """
    import gc

    gc.collect()
    try:
        return gui.GeorefGuiApp(pdf_path=pdf_path, output_dir=output_dir)
    except tk.TclError:
        gc.collect()
        return gui.GeorefGuiApp(pdf_path=pdf_path, output_dir=output_dir)


@contextmanager
def _app_context(pdf_path: Path, output_dir: Path):
    import gc
    import time

    # Force cleanup of any prior Tk root so the next creation lands on a
    # healthy Tcl interpreter. On Windows, rapid create/destroy cycles leave
    # the interpreter in a partially initialised state (msgcat/tk_library
    # failures); gc + a brief idle give Tcl time to settle.
    gc.collect()

    app = _create_app(pdf_path, output_dir)
    app.update_idletasks()
    try:
        yield app
    finally:
        try:
            app.update_idletasks()
            app.destroy()
        except tk.TclError:
            pass
        gc.collect()
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Click-to-model mapping (pure, no Tk needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flip_y", [True, False])
def test_click_to_model_inverts_display_matrix_for_both_y_directions(flip_y):
    page_width_pt, page_height_pt = 200.0, 100.0
    render_scale = 2.0  # 2 pixels per PDF point
    display_to_model = gui.build_display_to_model_matrix(
        page_width_pt=page_width_pt,
        page_height_pt=page_height_pt,
        render_scale=render_scale,
        flip_y=flip_y,
    )

    a, _b, _c, _d, _e, _f = display_to_model

    model_x = 25.4
    model_y = 12.7
    expected_px = a * model_x
    expected_py = (display_to_model[3] * model_y
                   + (display_to_model[5] if flip_y else 0.0))

    mx, my = gui.canvas_to_model(
        canvas_x=expected_px,
        canvas_y=expected_py,
        image_offset_x=0.0,
        image_offset_y=0.0,
        canvas_zoom=1.0,
        display_to_model=display_to_model,
    )

    assert mx == pytest.approx(model_x, abs=1e-9)
    assert my == pytest.approx(model_y, abs=1e-9)


@pytest.mark.parametrize("flip_y", [True, False])
def test_display_matrix_derives_unit_from_upstream_helper(flip_y):
    """The GUI must share the same PDF-point-to-mm constant as the upstream
    :func:`librecad_pdf_importer.raster_geometry.display_to_model_matrix`
    helper. This contract test pins that contract for both flip_y values."""

    from types import SimpleNamespace

    from librecad_pdf_importer.raster_geometry import display_to_model_matrix

    page_width_pt, page_height_pt = 200.0, 100.0
    render_scale = 2.0

    upstream = display_to_model_matrix(
        SimpleNamespace(height=page_height_pt), scale=1.0, flip_y=flip_y
    )
    mm_per_pt = upstream[0]  # 25.4 / 72 at scale=1.0
    expected_unit = render_scale / mm_per_pt

    matrix = gui.build_display_to_model_matrix(
        page_width_pt=page_width_pt,
        page_height_pt=page_height_pt,
        render_scale=render_scale,
        flip_y=flip_y,
    )

    assert matrix[0] == pytest.approx(expected_unit, abs=1e-12)
    assert matrix[1] == 0.0
    assert matrix[2] == 0.0
    assert matrix[3] == pytest.approx(-expected_unit if flip_y else expected_unit, abs=1e-12)
    assert matrix[4] == 0.0
    expected_f = page_height_pt * render_scale if flip_y else 0.0
    assert matrix[5] == pytest.approx(expected_f, abs=1e-9)


@pytest.mark.parametrize("flip_y", [True, False])
def test_click_to_model_respects_canvas_offset_and_zoom(flip_y):
    page_width_pt, page_height_pt = 100.0, 100.0
    render_scale = 1.0
    display_to_model = gui.build_display_to_model_matrix(
        page_width_pt=page_width_pt,
        page_height_pt=page_height_pt,
        render_scale=render_scale,
        flip_y=flip_y,
    )

    image_offset_x, image_offset_y = 13.0, 17.0
    canvas_zoom = 0.5

    # Click on canvas at (43, 47):
    #   image pixel = ((43 - 13) / 0.5, (47 - 17) / 0.5) = (60, 60)
    mx, my = gui.canvas_to_model(
        canvas_x=43.0,
        canvas_y=47.0,
        image_offset_x=image_offset_x,
        image_offset_y=image_offset_y,
        canvas_zoom=canvas_zoom,
        display_to_model=display_to_model,
    )
    a, _b, _c, d, _e, f = display_to_model
    assert mx == pytest.approx(60.0 / a, abs=1e-9)
    # matrix maps model -> image_pixel, so invert: my = (image_py - f) / d.
    if flip_y:
        assert my == pytest.approx((60.0 - f) / d, abs=1e-9)
    else:
        assert my == pytest.approx((60.0 - f) / d, abs=1e-9)


def test_click_to_model_rejects_non_positive_canvas_zoom():
    display_to_model = gui.build_display_to_model_matrix(
        page_width_pt=100.0, page_height_pt=100.0, render_scale=1.0, flip_y=True,
    )
    with pytest.raises(ValueError):
        gui.canvas_to_model(
            canvas_x=10.0, canvas_y=10.0,
            image_offset_x=0.0, image_offset_y=0.0,
            canvas_zoom=0.0, display_to_model=display_to_model,
        )


def test_click_to_model_rejects_invalid_render_scale():
    with pytest.raises(ValueError):
        gui.build_display_to_model_matrix(
            page_width_pt=100.0, page_height_pt=100.0, render_scale=0.0, flip_y=True,
        )


# ---------------------------------------------------------------------------
# State operations: add / remove / clear / load / save GCPs (pure)
# ---------------------------------------------------------------------------


def test_add_gcp_appends_with_sequential_label_when_blank():
    state = gui.new_state()
    state = gui.add_gcp(state, source_x=10.0, source_y=20.0, world_x=100.0, world_y=200.0)
    state = gui.add_gcp(state, source_x=15.0, source_y=25.0, world_x=110.0, world_y=210.0)
    assert len(state.gcps) == 2
    assert [gcp.label for gcp in state.gcps] == ["P1", "P2"]


def test_remove_gcp_drops_by_index_and_preserves_order():
    state = gui.new_state()
    state = gui.add_gcp(state, 1.0, 2.0, 3.0, 4.0, label="A")
    state = gui.add_gcp(state, 5.0, 6.0, 7.0, 8.0, label="B")
    state = gui.add_gcp(state, 9.0, 10.0, 11.0, 12.0, label="C")

    state = gui.remove_gcp(state, 1)

    assert [gcp.label for gcp in state.gcps] == ["A", "C"]


def test_clear_gcps_returns_empty_collection():
    state = gui.new_state()
    state = gui.add_gcp(state, 1.0, 2.0, 3.0, 4.0)
    state = gui.add_gcp(state, 5.0, 6.0, 7.0, 8.0)

    cleared = gui.clear_gcps(state)

    assert cleared.gcps == ()


def test_save_and_load_gcps_round_trip_uses_public_schema(tmp_path):
    state = gui.new_state()
    state = gui.add_gcp(state, 1.5, 2.5, 100.5, 200.5, label="Corner A")
    state = gui.add_gcp(state, -3.0, 4.0, 50.0, -75.0, label="Corner B")
    path = tmp_path / "drawing.gcps.json"

    gui.save_state_gcps(state, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["gcps"]) == 2
    assert payload["gcps"][0]["label"] == "Corner A"

    reloaded = gui.load_state_gcps(gui.new_state(), path)
    assert reloaded.gcps == state.gcps


# ---------------------------------------------------------------------------
# Live QC summary
# ---------------------------------------------------------------------------


def test_live_qc_is_idle_with_no_gcps():
    state = gui.new_state(transform="helmert")
    qc = gui.compute_live_qc(state)
    assert qc.status == "insufficient"
    assert qc.method is None
    assert qc.message == gui.MIN_HELMERT_GCPS_MESSAGE


def test_live_qc_stays_idle_with_one_helmert_gcp():
    state = gui.new_state(transform="helmert")
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    qc = gui.compute_live_qc(state)
    assert qc.status == "insufficient"


def test_live_qc_returns_scale_rotation_and_pass_for_clean_helmert():
    state = gui.new_state(transform="helmert", rmse_threshold=0.5)
    source_points = ((10.0, 10.0), (30.0, 10.0), (10.0, 30.0))
    for sx, sy in source_points:
        wx = 2.0 * sx + 100.0
        wy = 2.0 * sy - 50.0
        state = gui.add_gcp(state, sx, sy, wx, wy)
    qc = gui.compute_live_qc(state)
    assert qc.status == "solved"
    assert qc.method == "helmert"
    assert qc.scale == pytest.approx(2.0, abs=1e-9)
    assert qc.rotation_deg == pytest.approx(0.0, abs=1e-9)
    assert qc.rmse == pytest.approx(0.0, abs=1e-12)
    assert qc.threshold_status == "pass"


def test_live_qc_flags_warning_when_rmse_above_threshold():
    state = gui.new_state(transform="helmert", rmse_threshold=0.01)
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    state = gui.add_gcp(state, 10.0, 0.0, 20.0, 0.0)
    state = gui.add_gcp(state, 0.0, 10.0, 0.0, 20.0)
    state = gui.add_gcp(state, 10.0, 10.0, 20.5, 20.0)  # noise
    qc = gui.compute_live_qc(state)
    assert qc.status == "solved"
    assert qc.threshold_status == "warning"
    assert qc.max_residual is not None and qc.max_residual > 0.01


def test_live_qc_keeps_insufficient_until_three_affine_gcps():
    state = gui.new_state(transform="affine")
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    state = gui.add_gcp(state, 1.0, 0.0, 2.0, 0.0)
    qc = gui.compute_live_qc(state)
    assert qc.status == "insufficient"
    state = gui.add_gcp(state, 0.0, 1.0, 0.0, 2.0)
    qc = gui.compute_live_qc(state)
    assert qc.status == "solved"


def test_live_qc_returns_error_for_rank_deficient_layout():
    state = gui.new_state(transform="helmert")
    state = gui.add_gcp(state, 0.0, 0.0, 1.0, 1.0)
    state = gui.add_gcp(state, 0.0, 0.0, 2.0, 2.0)  # duplicate source
    qc = gui.compute_live_qc(state)
    assert qc.status == "error"
    assert qc.method == "helmert"


# ---------------------------------------------------------------------------
# Export gate
# ---------------------------------------------------------------------------


def test_export_gate_requires_pdf_output_dir_and_sufficient_gcps(tmp_path):
    source = tmp_path / "drawing.pdf"
    _write_vector_pdf(source)

    state = gui.new_state(pdf_path=source, output_dir=tmp_path)
    enabled, reason = gui.export_gate(state)
    assert not enabled
    assert "GCP" in reason

    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    enabled, reason = gui.export_gate(state)
    assert not enabled

    state = gui.add_gcp(state, 10.0, 0.0, 20.0, 0.0)
    enabled, reason = gui.export_gate(state)
    assert enabled
    assert reason == ""


def test_export_gate_rejects_missing_pdf(tmp_path):
    state = gui.new_state(
        pdf_path=tmp_path / "missing.pdf", output_dir=tmp_path, transform="helmert",
    )
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    state = gui.add_gcp(state, 10.0, 0.0, 20.0, 0.0)
    enabled, reason = gui.export_gate(state)
    assert not enabled
    assert "PDF" in reason or "input" in reason.lower()


def test_export_gate_rejects_insufficient_for_affine(tmp_path):
    source = tmp_path / "drawing.pdf"
    _write_vector_pdf(source)
    state = gui.new_state(pdf_path=source, output_dir=tmp_path, transform="affine")
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    state = gui.add_gcp(state, 10.0, 0.0, 20.0, 0.0)
    enabled, reason = gui.export_gate(state)
    assert not enabled
    assert "3" in reason or "affine" in reason.lower()


def test_export_gate_rejects_duplicate_source_points_for_helmert(tmp_path):
    """A duplicate source point makes Helmert rank-deficient; export must
    surface the solver message instead of the GCP-count message."""
    source = tmp_path / "drawing.pdf"
    _write_vector_pdf(source)
    state = gui.new_state(pdf_path=source, output_dir=tmp_path, transform="helmert")
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    state = gui.add_gcp(state, 0.0, 0.0, 10.0, 10.0)  # duplicate source
    enabled, reason = gui.export_gate(state)
    assert enabled is False
    assert "rank-deficient" in reason.lower() or "duplicate" in reason.lower()


def test_export_gate_rejects_collinear_sources_for_affine(tmp_path):
    """Three collinear source points collapse affine; export must surface
    the solver message instead of the GCP-count message."""
    source = tmp_path / "drawing.pdf"
    _write_vector_pdf(source)
    state = gui.new_state(pdf_path=source, output_dir=tmp_path, transform="affine")
    # Three source points on the x-axis: (0,0), (10,0), (20,0) -> rank-deficient
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    state = gui.add_gcp(state, 10.0, 0.0, 5.0, 5.0)
    state = gui.add_gcp(state, 20.0, 0.0, 10.0, 10.0)
    enabled, reason = gui.export_gate(state)
    assert enabled is False
    assert "rank-deficient" in reason.lower() or "collinear" in reason.lower()


def test_export_gate_still_enables_for_solvable_helmert(tmp_path):
    """A well-conditioned Helmert layout must still pass the gate."""
    source = tmp_path / "drawing.pdf"
    _write_vector_pdf(source)
    state = gui.new_state(pdf_path=source, output_dir=tmp_path, transform="helmert")
    state = gui.add_gcp(state, 0.0, 0.0, 0.0, 0.0)
    state = gui.add_gcp(state, 10.0, 0.0, 20.0, 0.0)
    enabled, reason = gui.export_gate(state)
    assert enabled is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Tkinter integration (headless)
# ---------------------------------------------------------------------------


def _build_app(tmp_path):
    source = tmp_path / "drawing.pdf"
    _write_vector_pdf(source)
    output_dir = tmp_path / "out"
    return source, output_dir


def test_app_loads_first_page_and_keeps_model_units(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        assert app.state.pdf_path == source
        a, _b, _c, d, _e, f = app.display_to_model
        # a = render_scale / (25.4/72) = render_scale * 72/25.4
        expected_a = gui.DEFAULT_RENDER_SCALE * 72.0 / 25.4
        assert a == pytest.approx(expected_a)
        assert d < 0  # flip_y=True default
        assert f > 0


def test_pdf_path_field_is_readonly_and_tracks_the_rendered_source(tmp_path):
    source, output_dir = _build_app(tmp_path)

    with _app_context(source, output_dir) as app:
        assert str(app._entry_pdf["state"]) == "readonly"
        assert app._var_pdf.get() == str(source)


def test_opening_a_different_pdf_clears_source_bound_gcps(tmp_path):
    source, output_dir = _build_app(tmp_path)
    replacement = tmp_path / "replacement.pdf"
    _write_vector_pdf(replacement)

    with _app_context(source, output_dir) as app:
        app.state = gui.add_gcp(app.state, 0.0, 0.0, 0.0, 0.0)
        app.state = gui.add_gcp(app.state, 10.0, 0.0, 20.0, 0.0)
        app._refresh_after_state_change()
        assert str(app._btn_export["state"]) in ("normal", "!disabled")

        app._open_pdf(replacement)

        assert app.state.pdf_path == replacement
        assert app.state.gcps == ()
        assert app.tree.get_children() == ()
        assert str(app._btn_export["state"]) == "disabled"


def test_click_in_canvas_adds_gcp_with_model_coordinates(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        a, _b, _c, d, _e, f = app.display_to_model
        target_model = (12.7, 6.35)
        target_px_x = a * target_model[0] + _c * target_model[1] + _e
        target_px_y = _b * target_model[0] + d * target_model[1] + f
        cx = target_px_x * app.canvas_zoom + app.image_offset_x
        cy = target_px_y * app.canvas_zoom + app.image_offset_y

        app._handle_canvas_click(
            canvas_x=cx, canvas_y=cy,
            world_x=50.0, world_y=60.0, label="Anchor 1",
        )

        assert len(app.state.gcps) == 1
        gcp = app.state.gcps[0]
        assert gcp.source_x == pytest.approx(target_model[0], abs=1e-6)
        assert gcp.source_y == pytest.approx(target_model[1], abs=1e-6)
        assert gcp.world_x == 50.0
        assert gcp.world_y == 60.0
        assert gcp.label == "Anchor 1"


def test_export_button_is_disabled_until_solvable(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        app.state = gui.add_gcp(app.state, 0.0, 0.0, 0.0, 0.0)
        app.state = gui.add_gcp(app.state, 10.0, 0.0, 20.0, 0.0)
        app._refresh_export_gate()
        assert str(app._btn_export["state"]) in ("normal", "!disabled")

        app.state = gui.remove_gcp(app.state, 1)
        app._refresh_export_gate()
        assert str(app._btn_export["state"]) == "disabled"


def test_browsing_output_refreshes_the_export_gate_immediately(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        app.state = gui.add_gcp(app.state, 0.0, 0.0, 0.0, 0.0)
        app.state = gui.add_gcp(app.state, 10.0, 0.0, 20.0, 0.0)
        app.state = gui.replace(app.state, output_dir=None)
        app._var_output.set("")
        app._refresh_after_state_change()
        assert str(app._btn_export["state"]) == "disabled"

        with patch.object(gui.filedialog, "askdirectory", return_value=str(output_dir)):
            app._browse_output()

        assert app.state.output_dir == output_dir
        assert str(app._btn_export["state"]) in ("normal", "!disabled")


def test_job_control_changes_refresh_state_qc_and_export_gate(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        app.state = gui.add_gcp(app.state, 0.0, 0.0, 0.0, 0.0)
        app.state = gui.add_gcp(app.state, 10.0, 0.0, 20.0, 0.0)
        app._refresh_after_state_change()
        assert str(app._btn_export["state"]) in ("normal", "!disabled")

        app._var_transform.set("affine")
        assert app.state.transform == "affine"
        assert str(app._btn_export["state"]) == "disabled"

        app._var_transform.set("helmert")
        app._var_crs.set("EPSG:5186")
        app._var_threshold.set("0.25")
        app._var_dxf.set("R2013")
        app._var_include_text.set(False)
        app._var_text_mode.set("geometry")

        assert app.state.transform == "helmert"
        assert app.state.crs == "EPSG:5186"
        assert app.state.rmse_threshold == pytest.approx(0.25)
        assert app.state.dxf_version == "R2013"
        assert app.state.include_text is False
        assert app.state.text_mode == "geometry"
        assert "Threshold:   0.25" in app.qc_text.get("1.0", tk.END)
        assert str(app._btn_export["state"]) in ("normal", "!disabled")


def test_remove_button_drops_selected_gcp_from_treeview(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        app.state = gui.add_gcp(app.state, 0.0, 0.0, 0.0, 0.0, label="A")
        app.state = gui.add_gcp(app.state, 1.0, 0.0, 2.0, 0.0, label="B")
        app._refresh_gcp_table()
        app.tree.selection_set(app.tree.get_children()[0])
        app._remove_selected_gcp()
        assert [gcp.label for gcp in app.state.gcps] == ["B"]


def test_export_runs_pipeline_asynchronously_and_reports_paths(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        app.state = gui.add_gcp(app.state, 5.0, 5.0, 10.0, 10.0)
        app.state = gui.add_gcp(app.state, 25.4, 5.0, 50.8, 10.0)
        app._refresh_export_gate()
        captured = SimpleNamespace(ran=False, kwargs=None)

        def fake_run(pdf_path, gcp_path, **kwargs):
            captured.ran = True
            captured.kwargs = dict(kwargs)
            save_gcps(gcp_path, app.state.gcps)
            return SimpleNamespace(
                dxf_path=kwargs["output_dir"] / "drawing_georef.dxf",
                json_path=kwargs["output_dir"] / "drawing_georef.json",
                html_path=kwargs["output_dir"] / "drawing_georef_report.html",
                report={"rmse": 0.0, "max_residual": 0.0, "threshold_status": "pass"},
            )

        with patch.object(gui, "run_georef_pipeline", side_effect=fake_run) as runner, \
             patch.object(gui.messagebox, "showinfo") as infobox, \
             patch.object(gui.messagebox, "showerror") as errorbox:
            app._start_export()
            app._export_done_event.wait(timeout=5.0)
            app._drain_export_queue()

        assert captured.ran is True
        assert captured.kwargs["transform"] == app.state.transform
        assert captured.kwargs["crs"] == app.state.crs
        assert captured.kwargs["dxf_version"] in gui.GUI_DXF_VERSIONS
        assert captured.kwargs["rmse_threshold"] == app.state.rmse_threshold
        assert captured.kwargs["include_text"] == app.state.include_text
        assert captured.kwargs["text_mode"] == app.state.text_mode
        runner.assert_called_once()
        assert infobox.called
        assert errorbox.called is False


def test_export_surfaces_error_message_on_pipeline_failure(tmp_path):
    source, output_dir = _build_app(tmp_path)
    with _app_context(source, output_dir) as app:
        app.state = gui.add_gcp(app.state, 0.0, 0.0, 0.0, 0.0)
        app.state = gui.add_gcp(app.state, 10.0, 0.0, 20.0, 0.0)
        app._refresh_export_gate()

        def fail(*_a, **_k):
            raise RuntimeError("simulated export failure")

        with patch.object(gui, "run_georef_pipeline", side_effect=fail), \
             patch.object(gui.messagebox, "showerror") as errorbox:
            app._start_export()
            app._export_done_event.wait(timeout=5.0)
            app._drain_export_queue()
        assert errorbox.called
        # messagebox.showerror(title, message, ...) -> message is args[1]
        assert "simulated export failure" in errorbox.call_args.args[1]


# ---------------------------------------------------------------------------
# Registration / wiring
# ---------------------------------------------------------------------------


def test_module_exposes_main_callable_for_entry_point():
    assert callable(gui.main)


def test_main_self_test_uses_runtime_probe_without_constructing_tk(monkeypatch):
    from librecad_pdf_importer import runtime_self_test

    calls = []
    monkeypatch.setattr(
        runtime_self_test,
        "run_runtime_self_test",
        lambda: calls.append("self-test") or 0,
    )
    monkeypatch.setattr(
        gui,
        "GeorefGuiApp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Tk started")),
    )

    assert gui.main(["--self-test"]) == 0
    assert calls == ["self-test"]


def test_main_returns_zero_when_no_event_loop_is_started(monkeypatch):
    """Headless smoke: ensure main() constructs the app and exits cleanly."""
    calls: list[str] = []

    class _FakeApp:
        def __init__(self, *args, **kwargs) -> None:
            calls.append("init")

        def mainloop(self) -> None:
            calls.append("mainloop")

        def withdraw(self) -> None:
            calls.append("withdraw")

        def destroy(self) -> None:
            calls.append("destroy")

    monkeypatch.setattr(gui, "GeorefGuiApp", _FakeApp)
    assert gui.main(["dummy.pdf"]) == 0
    assert calls[:3] == ["init", "mainloop", "destroy"]
    assert gui.main([]) == 0
    result = gui.main()
    assert result == 0
    assert "mainloop" in calls
