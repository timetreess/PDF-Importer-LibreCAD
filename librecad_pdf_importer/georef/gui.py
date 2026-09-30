"""Tkinter GCP-picking GUI for PDF2GeoCAD.

This module owns the small industrial desktop interface used to register
ground-control points on a single PDF page and export a real-coordinate DXF
through :func:`librecad_pdf_importer.georef.pipeline.run_georef_pipeline`.

Conversion logic lives in the georef package. The GUI is responsible for
mapping pixel clicks back into the same model space used by extraction,
exposing a ttk-style control surface, and asynchronously running the public
pipeline so the desktop remains responsive while exports run.
"""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Sequence, Tuple

import pymupdf as fitz

from librecad_pdf_importer.georef.gcp import load_gcps, save_gcps
from librecad_pdf_importer.georef.models import (
    GCP,
    GeoreferenceError,
    TransformResult,
)
from librecad_pdf_importer.georef.pipeline import (
    GEOREF_DXF_VERSIONS,
    run_georef_pipeline,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


GUI_DXF_VERSIONS = GEOREF_DXF_VERSIONS
DEFAULT_DXF_VERSION = "R2018"
DEFAULT_TRANSFORM = "helmert"
DEFAULT_TEXT_MODE = "text"
TEXT_MODES: tuple[str, ...] = ("text", "labels", "3d_text", "glyphs", "geometry")
MIN_HELMERT_GCPS_MESSAGE = "Add at least 2 GCPs for Helmert"
MIN_AFFINE_GCPS_MESSAGE = "Add at least 3 GCPs for affine"
DEFAULT_RENDER_SCALE = 2.0  # 144 DPI at 72 DPI base
DEFAULT_CANVAS_ZOOM = 1.0
MAX_CANVAS_ZOOM = 4.0
MIN_CANVAS_ZOOM = 0.25


# ---------------------------------------------------------------------------
# Display / click mapping
# ---------------------------------------------------------------------------


def build_display_to_model_matrix(
    *,
    page_width_pt: float,
    page_height_pt: float,
    render_scale: float,
    flip_y: bool,
) -> Tuple[float, float, float, float, float, float]:
    """Return the 6-tuple affine that maps model mm -> image pixels. It is the
    inverse of :func:`librecad_pdf_importer.raster_geometry.display_to_model_matrix`
    (which maps pixel -> mm) scaled by ``render_scale`` (pixels per PDF point),
    so the GUI shares the same PDF-point-to-mm constant as extraction.

    ``flip_y=True`` matches PyMuPDF's top-to-bottom rendering: the Y component
    flips and shifts by the page height in pixels. ``flip_y=False`` keeps Y
    pointing up, matching the model space used by the georeferencing solver.
    """

    from types import SimpleNamespace

    from librecad_pdf_importer.raster_geometry import display_to_model_matrix

    if page_width_pt <= 0 or page_height_pt <= 0:
        raise ValueError("Page dimensions must be positive")
    if render_scale <= 0 or not math.isfinite(render_scale):
        raise ValueError("Render scale must be a finite positive number")

    # Pull the mm-per-PDF-point constant from the upstream helper so the GUI
    # never silently drifts from the extraction-side conversion.
    page_rect = SimpleNamespace(height=float(page_height_pt))
    pixel_to_mm = display_to_model_matrix(page_rect, scale=1.0, flip_y=bool(flip_y))
    mm_per_pt = pixel_to_mm[0]
    if not math.isfinite(mm_per_pt) or mm_per_pt <= 0:
        raise ValueError("Upstream page mapping produced a non-positive unit")

    # model_mm -> pt: divide by mm_per_pt. pt -> pixel: multiply by render_scale.
    a = float(render_scale) / mm_per_pt
    d = -a if bool(flip_y) else a
    e = 0.0
    f = float(page_height_pt) * float(render_scale) if bool(flip_y) else 0.0
    return (a, 0.0, 0.0, d, e, f)


def canvas_to_model(
    *,
    canvas_x: float,
    canvas_y: float,
    image_offset_x: float,
    image_offset_y: float,
    canvas_zoom: float,
    display_to_model: Sequence[float],
) -> Tuple[float, float]:
    """Map a canvas pixel position back into model mm coordinates."""

    if len(display_to_model) != 6:
        raise ValueError("Display-to-model matrix must have six coefficients")
    if canvas_zoom <= 0 or not math.isfinite(canvas_zoom):
        raise ValueError("Canvas zoom must be a finite positive number")
    a, b, c, d, e, f = (float(value) for value in display_to_model)
    if a * d - b * c == 0:
        raise ValueError("Display-to-model matrix is singular")

    image_px = (canvas_x - float(image_offset_x)) / float(canvas_zoom)
    image_py = (canvas_y - float(image_offset_y)) / float(canvas_zoom)

    # Invert the 2x2 linear part. With the production matrix this is purely
    # diagonal so a closed form is safe and avoids numpy in the click path.
    det = a * d - b * c
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det

    mx = inv_a * (image_px - e) + inv_c * (image_py - f)
    my = inv_b * (image_px - e) + inv_d * (image_py - f)
    return float(mx), float(my)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeorefGuiState:
    """Pure, immutable view of every GUI-controllable parameter and the GCPs."""

    pdf_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    transform: str = DEFAULT_TRANSFORM
    crs: str = "local"
    rmse_threshold: float = 1.0
    dxf_version: str = DEFAULT_DXF_VERSION
    include_text: bool = True
    text_mode: str = DEFAULT_TEXT_MODE
    gcps: Tuple[GCP, ...] = field(default_factory=tuple)


def new_state(
    *,
    pdf_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    transform: str = DEFAULT_TRANSFORM,
    crs: str = "local",
    rmse_threshold: float = 1.0,
    dxf_version: str = DEFAULT_DXF_VERSION,
    include_text: bool = True,
    text_mode: str = DEFAULT_TEXT_MODE,
) -> GeorefGuiState:
    """Build a state with normalized string options."""

    normalized_transform = str(transform).strip().lower()
    if normalized_transform not in {"helmert", "affine"}:
        raise ValueError("Transform must be 'helmert' or 'affine'")
    normalized_dxf = str(dxf_version).strip().upper()
    if normalized_dxf not in GUI_DXF_VERSIONS:
        raise ValueError("DXF version must be one of the supported georef versions")
    normalized_text_mode = str(text_mode).strip().lower()
    if normalized_text_mode not in TEXT_MODES:
        raise ValueError("Text mode must be one of the supported representations")
    threshold = float(rmse_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("RMSE threshold must be a finite non-negative number")
    return GeorefGuiState(
        pdf_path=Path(pdf_path) if pdf_path is not None else None,
        output_dir=Path(output_dir) if output_dir is not None else None,
        transform=normalized_transform,
        crs=str(crs).strip() or "local",
        rmse_threshold=threshold,
        dxf_version=normalized_dxf,
        include_text=bool(include_text),
        text_mode=normalized_text_mode,
        gcps=(),
    )


def add_gcp(
    state: GeorefGuiState,
    source_x: float,
    source_y: float,
    world_x: float,
    world_y: float,
    label: str = "",
) -> GeorefGuiState:
    """Append a GCP and auto-label it ``P<n>`` when ``label`` is blank."""

    gcp = GCP(
        source_x=float(source_x),
        source_y=float(source_y),
        world_x=float(world_x),
        world_y=float(world_y),
        label=str(label) if label else f"P{len(state.gcps) + 1}",
    )
    return replace(state, gcps=state.gcps + (gcp,))


def remove_gcp(state: GeorefGuiState, index: int) -> GeorefGuiState:
    """Drop the GCP at ``index`` (0-based)."""

    if index < 0 or index >= len(state.gcps):
        raise IndexError(f"GCP index out of range: {index}")
    remaining = state.gcps[:index] + state.gcps[index + 1:]
    return replace(state, gcps=remaining)


def clear_gcps(state: GeorefGuiState) -> GeorefGuiState:
    return replace(state, gcps=())


def load_state_gcps(state: GeorefGuiState, path: Path) -> GeorefGuiState:
    """Replace the state's GCPs from a JSON file."""

    loaded = load_gcps(path)
    return replace(state, gcps=tuple(loaded))


def save_state_gcps(state: GeorefGuiState, path: Path) -> Path:
    """Write the state's GCPs using the production GCP JSON schema."""

    save_gcps(path, state.gcps)
    return Path(path)


# ---------------------------------------------------------------------------
# Live solver summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveQC:
    """Renderable summary of the current GCPs under the chosen transform."""

    status: str  # "insufficient", "solved", or "error"
    method: Optional[str] = None
    scale: Optional[float] = None
    rotation_deg: Optional[float] = None
    rmse: Optional[float] = None
    max_residual: Optional[float] = None
    threshold_status: Optional[str] = None  # "pass", "warning", or None
    message: str = ""


def compute_live_qc(state: GeorefGuiState) -> LiveQC:
    """Solve the current GCPs for the GUI's chosen transform, if possible."""

    required = 2 if state.transform == "helmert" else 3
    if len(state.gcps) < required:
        message = (
            MIN_HELMERT_GCPS_MESSAGE
            if state.transform == "helmert"
            else MIN_AFFINE_GCPS_MESSAGE
        )
        return LiveQC(status="insufficient", message=message)

    try:
        result: TransformResult = (
            _solve_helmert(state.gcps) if state.transform == "helmert" else _solve_affine(state.gcps)
        )
    except GeoreferenceError as exc:
        return LiveQC(status="error", method=state.transform, message=str(exc))

    threshold_status = "pass" if result.rmse <= state.rmse_threshold else "warning"
    return LiveQC(
        status="solved",
        method=result.method,
        scale=result.scale,
        rotation_deg=result.rotation_deg,
        rmse=result.rmse,
        max_residual=result.max_residual,
        threshold_status=threshold_status,
        message="",
    )


def _solve_helmert(gcps):
    from librecad_pdf_importer.georef.helmert import solve_helmert
    return solve_helmert(gcps)


def _solve_affine(gcps):
    from librecad_pdf_importer.georef.affine import solve_affine
    return solve_affine(gcps)


# ---------------------------------------------------------------------------
# Export gate
# ---------------------------------------------------------------------------


def export_gate(state: GeorefGuiState) -> Tuple[bool, str]:
    """Return (enabled, reason). The button is only enabled when the entire
    pipeline can succeed: a PDF, an output directory, and enough GCPs that the
    chosen solver actually accepts (rank-sufficient, no duplicate points)."""

    if state.pdf_path is None or not state.pdf_path.is_file():
        return False, "Select an input PDF"
    if state.output_dir is None:
        return False, "Select an output directory"
    qc = compute_live_qc(state)
    if qc.status == "insufficient":
        required = 2 if state.transform == "helmert" else 3
        required_label = "2 GCPs" if required == 2 else "3 GCPs"
        return False, f"Need {required_label} for {state.transform} (have {len(state.gcps)})"
    if qc.status == "error":
        return False, qc.message or f"{state.transform} solver rejected the GCP layout"
    return True, ""


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedPage:
    """The PyMuPDF-derived image displayed on the canvas."""

    page: object
    page_width_pt: float
    page_height_pt: float
    image_width_px: int
    image_height_px: int
    photo: object  # kept alive by the Tk app; type: tk.PhotoImage
    pil_image: object  # kept for optional save/export


def render_first_page(pdf_path: Path, *, render_scale: float) -> RenderedPage:
    """Render page 1 of ``pdf_path`` with PyMuPDF and return a Tk image."""

    from PIL import Image, ImageTk  # local import keeps optional dep light

    document = fitz.open(str(pdf_path))
    try:
        if document.page_count < 1:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        page = document.load_page(0)
        matrix = fitz.Matrix(float(render_scale), float(render_scale))
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGB" if pixmap.n < 4 else "RGBA"
        pil_image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        photo = ImageTk.PhotoImage(pil_image)
        return RenderedPage(
            page=page,
            page_width_pt=float(page.rect.width),
            page_height_pt=float(page.rect.height),
            image_width_px=int(pixmap.width),
            image_height_px=int(pixmap.height),
            photo=photo,
            pil_image=pil_image,
        )
    finally:
        document.close()


# ---------------------------------------------------------------------------
# Tkinter application
# ---------------------------------------------------------------------------


class _WorldEntryDialog(tk.Toplevel):
    """Modal dialog that asks the user for world X / Y / label."""

    def __init__(self, master: tk.Misc, *, default_label: str) -> None:
        super().__init__(master)
        self.title("GCP World Coordinates")
        self.transient(master)
        self.resizable(False, False)
        self.result: Optional[Tuple[float, float, str]] = None

        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="World X:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self._var_x = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self._var_x, width=18).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(frame, text="World Y:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self._var_y = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self._var_y, width=18).grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(frame, text="Label:").grid(row=2, column=0, sticky="e", padx=4, pady=2)
        self._var_label = tk.StringVar(value=default_label)
        ttk.Entry(frame, textvariable=self._var_label, width=18).grid(row=2, column=1, padx=4, pady=2)

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(buttons, text="OK", command=self._on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, padx=4)

        self.bind("<Return>", lambda _event: self._on_ok())
        self.bind("<Escape>", lambda _event: self._on_cancel())
        self.grab_set()
        self.focus_set()

    def _on_ok(self) -> None:
        try:
            x = float(self._var_x.get())
            y = float(self._var_y.get())
        except ValueError:
            messagebox.showerror("Invalid world coordinates", "World X and Y must be numbers.", parent=self)
            return
        self.result = (x, y, self._var_label.get().strip())
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


class GeorefGuiApp(tk.Tk):
    """The PDF2GeoCAD desktop GCP-picker.

    The application is a Tk root window. It exposes its current :class:`GeorefGuiState`
    and helpers as attributes so headless tests can exercise the click-to-model
    mapping and pipeline integration without driving Tk events.
    """

    def __init__(
        self,
        *args,
        pdf_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        render_scale: float = DEFAULT_RENDER_SCALE,
        flip_y: bool = True,
    ) -> None:
        super().__init__(*args)
        self.title("PDF2GeoCAD - Manual GCP Georeferencing")
        self.minsize(820, 560)
        self.geometry("980x640")

        self.state: GeorefGuiState = new_state(pdf_path=pdf_path, output_dir=output_dir)
        self.render_scale: float = float(render_scale)
        self.flip_y: bool = bool(flip_y)
        self.canvas_zoom: float = DEFAULT_CANVAS_ZOOM
        self.image_offset_x: float = 0.0
        self.image_offset_y: float = 0.0
        self.display_to_model: Tuple[float, ...] = ()
        self.rendered_page: Optional[RenderedPage] = None

        self._export_thread: Optional[threading.Thread] = None
        self._export_done_event = threading.Event()

        self._build_ui()
        self._refresh_state_controls()
        self._bind_job_control_updates()
        if pdf_path is not None:
            self._open_pdf(Path(pdf_path))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        try:
            style = ttk.Style(self)
        except tk.TclError:
            style = ttk.Style()
        # Industrial/utilitarian Tk theme with high-contrast text.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=22)
        style.configure("Status.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))

        pad = {"padx": 6, "pady": 3}
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        outer.columnconfigure(1, weight=3)
        outer.columnconfigure(2, weight=2)
        outer.rowconfigure(1, weight=1)

        # ---- Title row ----
        ttk.Label(outer, text="PDF2GeoCAD — Manual GCP Georeferencing", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6),
        )

        # ---- Left column: file / option controls ----
        left = ttk.LabelFrame(outer, text="Job", padding=6)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=4)

        ttk.Label(left, text="Input PDF:").grid(row=0, column=0, sticky="e", **pad)
        self._var_pdf = tk.StringVar()
        self._entry_pdf = ttk.Entry(
            left,
            textvariable=self._var_pdf,
            width=30,
            state="readonly",
        )
        self._entry_pdf.grid(
            row=0, column=1, sticky="ew", **pad,
        )
        ttk.Button(left, text="Browse…", command=self._browse_pdf).grid(
            row=0, column=2, **pad,
        )

        ttk.Label(left, text="Output dir:").grid(row=1, column=0, sticky="e", **pad)
        self._var_output = tk.StringVar()
        ttk.Entry(left, textvariable=self._var_output, width=30).grid(
            row=1, column=1, sticky="ew", **pad,
        )
        ttk.Button(left, text="Browse…", command=self._browse_output).grid(
            row=1, column=2, **pad,
        )

        ttk.Label(left, text="Transform:").grid(row=2, column=0, sticky="e", **pad)
        self._var_transform = tk.StringVar(value=DEFAULT_TRANSFORM)
        ttk.Combobox(
            left, textvariable=self._var_transform,
            values=("helmert", "affine"), state="readonly", width=12,
        ).grid(row=2, column=1, columnspan=2, sticky="w", **pad)

        ttk.Label(left, text="CRS / Mode:").grid(row=3, column=0, sticky="e", **pad)
        self._var_crs = tk.StringVar(value="local")
        ttk.Entry(left, textvariable=self._var_crs, width=15).grid(
            row=3, column=1, columnspan=2, sticky="w", **pad,
        )

        ttk.Label(left, text="RMSE threshold:").grid(row=4, column=0, sticky="e", **pad)
        self._var_threshold = tk.StringVar(value="1.0")
        ttk.Entry(left, textvariable=self._var_threshold, width=10).grid(
            row=4, column=1, sticky="w", **pad,
        )

        ttk.Label(left, text="DXF version:").grid(row=5, column=0, sticky="e", **pad)
        self._var_dxf = tk.StringVar(value=DEFAULT_DXF_VERSION)
        ttk.Combobox(
            left, textvariable=self._var_dxf,
            values=GUI_DXF_VERSIONS, state="readonly", width=10,
        ).grid(row=5, column=1, sticky="w", **pad)

        self._var_include_text = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            left, text="Include text", variable=self._var_include_text,
        ).grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(left, text="Text mode:").grid(row=7, column=0, sticky="e", **pad)
        self._var_text_mode = tk.StringVar(value=DEFAULT_TEXT_MODE)
        ttk.Combobox(
            left, textvariable=self._var_text_mode,
            values=TEXT_MODES, state="readonly", width=12,
        ).grid(row=7, column=1, sticky="w", **pad)

        # GCP list management
        gcp_buttons = ttk.LabelFrame(left, text="GCPs", padding=4)
        gcp_buttons.grid(row=8, column=0, columnspan=3, sticky="ew", padx=2, pady=(8, 4))
        ttk.Button(gcp_buttons, text="Load…", command=self._load_gcps_dialog).grid(row=0, column=0, padx=2)
        ttk.Button(gcp_buttons, text="Save…", command=self._save_gcps_dialog).grid(row=0, column=1, padx=2)
        ttk.Button(gcp_buttons, text="Remove", command=self._remove_selected_gcp).grid(row=0, column=2, padx=2)
        ttk.Button(gcp_buttons, text="Clear", command=self._clear_gcps).grid(row=0, column=3, padx=2)

        left.columnconfigure(1, weight=1)

        # ---- Middle column: PDF canvas with scrollbars ----
        middle = ttk.LabelFrame(outer, text="Page 1 (click to add GCP)", padding=4)
        middle.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        middle.columnconfigure(0, weight=1)
        middle.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(middle, background="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(middle, orient=tk.VERTICAL, command=self.canvas.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        xsb = ttk.Scrollbar(middle, orient=tk.HORIZONTAL, command=self.canvas.xview)
        xsb.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda _event: None)
        self._canvas_image_id: Optional[int] = None

        zoom_frame = ttk.Frame(middle)
        zoom_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(zoom_frame, text="Canvas zoom:").pack(side=tk.LEFT, padx=4)
        self._var_canvas_zoom = tk.StringVar(value=f"{DEFAULT_CANVAS_ZOOM:g}")
        ttk.Entry(zoom_frame, textvariable=self._var_canvas_zoom, width=6).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Apply", command=self._apply_canvas_zoom).pack(side=tk.LEFT, padx=4)

        # ---- Right column: GCP table + solver summary ----
        right = ttk.LabelFrame(outer, text="Solver", padding=4)
        right.grid(row=1, column=2, sticky="nsew", padx=(6, 0), pady=4)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        columns = ("index", "label", "source", "world")
        self.tree = ttk.Treeview(
            right, columns=columns, show="headings", height=12,
        )
        self.tree.heading("index", text="#")
        self.tree.heading("label", text="Label")
        self.tree.heading("source", text="Source X, Y (mm)")
        self.tree.heading("world", text="World X, Y")
        self.tree.column("index", width=32, anchor="e")
        self.tree.column("label", width=90, anchor="w")
        self.tree.column("source", width=110, anchor="w")
        self.tree.column("world", width=110, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.qc_text = tk.Text(right, height=10, width=1, wrap=tk.WORD, font=("Consolas", 9))
        self.qc_text.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self.qc_text.configure(state=tk.DISABLED)

        # ---- Bottom action bar ----
        actions = ttk.Frame(outer)
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=(6, 0))
        actions.columnconfigure(0, weight=1)
        self._status_label = ttk.Label(actions, text="Ready.", style="Status.TLabel")
        self._status_label.grid(row=0, column=0, sticky="w")
        self._btn_export = ttk.Button(actions, text="Export DXF", command=self._start_export)
        self._btn_export.grid(row=0, column=1, padx=4)
        self._btn_export.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # State sync helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self._status_label.configure(text=message)

    def _collect_state(self) -> GeorefGuiState:
        try:
            threshold = float(self._var_threshold.get())
        except ValueError:
            threshold = float("nan")
        # Run new_state() only to validate the settings, then merge into
        # the live state so the user's GCP list is preserved.
        validated = new_state(
            pdf_path=Path(self._var_pdf.get().strip()) if self._var_pdf.get().strip() else None,
            output_dir=Path(self._var_output.get().strip()) if self._var_output.get().strip() else None,
            transform=self._var_transform.get(),
            crs=self._var_crs.get().strip() or "local",
            rmse_threshold=threshold,
            dxf_version=self._var_dxf.get(),
            include_text=bool(self._var_include_text.get()),
            text_mode=self._var_text_mode.get(),
        )
        return replace(
            self.state,
            pdf_path=validated.pdf_path,
            output_dir=validated.output_dir,
            transform=validated.transform,
            crs=validated.crs,
            rmse_threshold=validated.rmse_threshold,
            dxf_version=validated.dxf_version,
            include_text=validated.include_text,
            text_mode=validated.text_mode,
        )

    def _refresh_state_controls(self) -> None:
        s = self.state
        self._var_pdf.set(str(s.pdf_path) if s.pdf_path else "")
        self._var_output.set(str(s.output_dir) if s.output_dir else "")
        self._var_transform.set(s.transform)
        self._var_crs.set(s.crs)
        self._var_threshold.set(f"{s.rmse_threshold:g}")
        self._var_dxf.set(s.dxf_version)
        self._var_include_text.set(s.include_text)
        self._var_text_mode.set(s.text_mode)

    def _bind_job_control_updates(self) -> None:
        for variable in (
            self._var_output,
            self._var_transform,
            self._var_crs,
            self._var_threshold,
            self._var_dxf,
            self._var_include_text,
            self._var_text_mode,
        ):
            variable.trace_add("write", self._on_job_control_changed)

    def _on_job_control_changed(self, *_args: str) -> None:
        try:
            self.state = self._collect_state()
        except ValueError as exc:
            self._btn_export.configure(state=tk.DISABLED)
            self._set_status(str(exc))
            return
        self._refresh_qc_panel()
        self._apply_export_gate()

    def _sync_state_from_controls(self) -> bool:
        try:
            self.state = self._collect_state()
        except ValueError as exc:
            messagebox.showerror("Invalid setting", str(exc))
            return False
        return True

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_pdf(self) -> None:
        initial = str(self.state.pdf_path.parent) if self.state.pdf_path else str(Path.cwd())
        path = filedialog.askopenfilename(
            parent=self, title="Select PDF file",
            initialdir=initial, filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self._open_pdf(Path(path))

    def _browse_output(self) -> None:
        initial = str(self.state.output_dir) if self.state.output_dir else str(Path.cwd())
        path = filedialog.askdirectory(parent=self, title="Select output directory", initialdir=initial)
        if path:
            self.state = replace(self.state, output_dir=Path(path))
            self._var_output.set(path)

    def _load_gcps_dialog(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Load GCP JSON",
            filetypes=[("GCP JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.state = load_state_gcps(self.state, Path(path))
        except GeoreferenceError as exc:
            messagebox.showerror("Cannot load GCPs", str(exc), parent=self)
            return
        self._refresh_after_state_change()

    def _save_gcps_dialog(self) -> None:
        if not self.state.gcps:
            messagebox.showinfo("No GCPs", "Add at least one GCP before saving.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Save GCP JSON",
            defaultextension=".json",
            filetypes=[("GCP JSON", "*.json")],
        )
        if not path:
            return
        try:
            save_state_gcps(self.state, Path(path))
        except GeoreferenceError as exc:
            messagebox.showerror("Cannot save GCPs", str(exc), parent=self)
            return
        self._set_status(f"Saved {len(self.state.gcps)} GCP(s) to {path}")

    # ------------------------------------------------------------------
    # PDF rendering
    # ------------------------------------------------------------------

    def _open_pdf(self, path: Path) -> None:
        try:
            rendered = render_first_page(path, render_scale=self.render_scale)
        except Exception as exc:  # noqa: BLE001 — surface any render failure
            messagebox.showerror("Cannot render PDF", str(exc), parent=self)
            return
        source_changed = self.state.pdf_path != path
        self.rendered_page = rendered
        self.state = replace(
            self.state,
            pdf_path=path,
            gcps=() if source_changed else self.state.gcps,
        )
        self._var_pdf.set(str(path))
        self.display_to_model = build_display_to_model_matrix(
            page_width_pt=rendered.page_width_pt,
            page_height_pt=rendered.page_height_pt,
            render_scale=self.render_scale,
            flip_y=self.flip_y,
        )
        self._draw_image()
        self._refresh_after_state_change()

    def _draw_image(self) -> None:
        if self.rendered_page is None:
            return
        self.canvas.delete("all")
        self._canvas_image_id = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.rendered_page.photo,
        )
        self.image_offset_x = 0.0
        self.image_offset_y = 0.0
        self.canvas_zoom = float(self._var_canvas_zoom.get() or DEFAULT_CANVAS_ZOOM)
        self.canvas.configure(
            scrollregion=(0, 0, self.rendered_page.image_width_px,
                          self.rendered_page.image_height_px),
        )

    def _apply_canvas_zoom(self) -> None:
        if self.rendered_page is None:
            return
        try:
            zoom = float(self._var_canvas_zoom.get())
        except ValueError:
            messagebox.showerror("Invalid zoom", "Zoom must be a positive number.", parent=self)
            return
        if not math.isfinite(zoom) or zoom <= 0:
            messagebox.showerror("Invalid zoom", "Zoom must be a positive number.", parent=self)
            return
        zoom = max(MIN_CANVAS_ZOOM, min(MAX_CANVAS_ZOOM, zoom))
        self.canvas_zoom = zoom
        self._var_canvas_zoom.set(f"{zoom:g}")
        # Re-scale the displayed image by resizing the PIL base via LANCZOS.
        base = self.rendered_page.pil_image
        from PIL import Image, ImageTk
        new_w = max(1, int(round(base.width * zoom)))
        new_h = max(1, int(round(base.height * zoom)))
        resampled = base.resize((new_w, new_h), Image.LANCZOS)
        self.rendered_page = RenderedPage(
            page=self.rendered_page.page,
            page_width_pt=self.rendered_page.page_width_pt,
            page_height_pt=self.rendered_page.page_height_pt,
            image_width_px=new_w,
            image_height_px=new_h,
            photo=ImageTk.PhotoImage(resampled),
            pil_image=base,
        )
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.rendered_page.photo)
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event: tk.Event) -> None:  # noqa: ARG002
        if self.rendered_page is None or not self.display_to_model:
            messagebox.showinfo("Open a PDF", "Open a PDF before adding GCPs.", parent=self)
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        try:
            sx, sy = canvas_to_model(
                canvas_x=canvas_x,
                canvas_y=canvas_y,
                image_offset_x=self.image_offset_x,
                image_offset_y=self.image_offset_y,
                canvas_zoom=self.canvas_zoom,
                display_to_model=self.display_to_model,
            )
        except ValueError as exc:
            messagebox.showerror("Click failed", str(exc), parent=self)
            return
        default_label = f"P{len(self.state.gcps) + 1}"
        dialog = _WorldEntryDialog(self, default_label=default_label)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        wx, wy, label = dialog.result
        self._handle_canvas_click(
            canvas_x=canvas_x, canvas_y=canvas_y,
            world_x=wx, world_y=wy, label=label,
        )

    def _handle_canvas_click(
        self,
        *,
        canvas_x: float,
        canvas_y: float,
        world_x: float,
        world_y: float,
        label: str,
    ) -> None:
        sx, sy = canvas_to_model(
            canvas_x=float(canvas_x),
            canvas_y=float(canvas_y),
            image_offset_x=self.image_offset_x,
            image_offset_y=self.image_offset_y,
            canvas_zoom=self.canvas_zoom,
            display_to_model=self.display_to_model,
        )
        self.state = add_gcp(
            self.state,
            source_x=sx, source_y=sy,
            world_x=world_x, world_y=world_y, label=label,
        )
        # Visual marker on the canvas.
        if self.rendered_page is not None and self._canvas_image_id is not None:
            self.canvas.create_oval(
                canvas_x - 4, canvas_y - 4, canvas_x + 4, canvas_y + 4,
                outline="#ffd400", width=2, fill="",
            )
            self.canvas.create_text(
                canvas_x + 6, canvas_y - 6,
                text=label or f"P{len(self.state.gcps)}",
                anchor=tk.NW, fill="#ffd400",
                font=("Segoe UI", 8, "bold"),
            )
        self._refresh_after_state_change()

    # ------------------------------------------------------------------
    # GCP table / summary
    # ------------------------------------------------------------------

    def _remove_selected_gcp(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        for item_id in selection:
            try:
                index = int(item_id)
            except ValueError:
                continue
            if 0 <= index < len(self.state.gcps):
                self.state = remove_gcp(self.state, index)
        self._refresh_after_state_change()

    def _clear_gcps(self) -> None:
        if not self.state.gcps:
            return
        if not messagebox.askyesno("Clear GCPs", "Remove all GCPs?", parent=self):
            return
        self.state = clear_gcps(self.state)
        self._refresh_after_state_change()

    def _refresh_after_state_change(self) -> None:
        self._refresh_gcp_table()
        self._refresh_qc_panel()
        self._refresh_export_gate()

    def _refresh_gcp_table(self) -> None:
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        for index, gcp in enumerate(self.state.gcps, start=1):
            self.tree.insert(
                "", "end", iid=str(index - 1),
                values=(
                    index,
                    gcp.label,
                    f"{gcp.source_x:.4f}, {gcp.source_y:.4f}",
                    f"{gcp.world_x:.4f}, {gcp.world_y:.4f}",
                ),
            )

    def _refresh_qc_panel(self) -> None:
        qc = compute_live_qc(self.state)
        lines = [
            f"Method:      {qc.method or '-'}",
            f"GCPs:        {len(self.state.gcps)}",
            f"Scale:       {qc.scale if qc.scale is not None else '-'}",
            f"Rotation:    {qc.rotation_deg if qc.rotation_deg is not None else '-'}",
            f"RMSE:        {qc.rmse if qc.rmse is not None else '-'}",
            f"Max resid.:  {qc.max_residual if qc.max_residual is not None else '-'}",
            f"Threshold:   {self.state.rmse_threshold:g}",
            f"Status:      {qc.threshold_status or qc.status}",
        ]
        if qc.message:
            lines.append(qc.message)
        self.qc_text.configure(state=tk.NORMAL)
        self.qc_text.delete("1.0", tk.END)
        self.qc_text.insert(tk.END, "\n".join(lines))
        self.qc_text.configure(state=tk.DISABLED)

    def _refresh_export_gate(self) -> None:
        # Pull latest control values into state first so gate reflects live edits.
        if not self._sync_state_from_controls():
            self._btn_export.configure(state=tk.DISABLED)
            return
        self._apply_export_gate()

    def _apply_export_gate(self) -> None:
        enabled, reason = export_gate(self.state)
        if enabled:
            self._btn_export.configure(state=tk.NORMAL)
            self._set_status(f"Ready to export ({len(self.state.gcps)} GCPs).")
        else:
            self._btn_export.configure(state=tk.DISABLED)
            self._set_status(reason)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _start_export(self) -> None:
        if not self._sync_state_from_controls():
            return
        enabled, reason = export_gate(self.state)
        if not enabled:
            messagebox.showwarning("Cannot export", reason, parent=self)
            return
        if self._export_thread is not None and self._export_thread.is_alive():
            return
        self._export_done_event.clear()
        self._export_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._btn_export.configure(state=tk.DISABLED)
        self._set_status("Exporting…")

        state_snapshot = self.state

        def worker() -> None:
            try:
                with _TemporaryGcpFile(state_snapshot) as gcp_path:
                    result = run_georef_pipeline(
                        state_snapshot.pdf_path,
                        gcp_path,
                        transform=state_snapshot.transform,
                        crs=state_snapshot.crs,
                        output_dir=state_snapshot.output_dir,
                        rmse_threshold=state_snapshot.rmse_threshold,
                        dxf_version=state_snapshot.dxf_version,
                        include_text=state_snapshot.include_text,
                        text_mode=state_snapshot.text_mode,
                    )
                self._export_queue.put(("success", result))
            except BaseException as exc:  # surface any failure via the queue
                self._export_queue.put(("error", exc))
            finally:
                self._export_done_event.set()

        self._export_thread = threading.Thread(target=worker, daemon=True)
        self._export_thread.start()
        # Poll the queue from the Tk main thread; safe to call before
        # mainloop because :meth:`after_idle` only schedules a callback.
        self.after(50, self._drain_export_queue)

    def _drain_export_queue(self) -> None:
        try:
            while True:
                kind, payload = self._export_queue.get_nowait()
                if kind == "success":
                    self._on_export_success(payload)
                else:
                    self._on_export_failure(payload)
        except (queue.Empty, AttributeError):
            pass
        if self._export_thread is not None and self._export_thread.is_alive():
            self.after(50, self._drain_export_queue)

    def _on_export_success(self, result) -> None:
        self._btn_export.configure(state=tk.NORMAL)
        self._set_status(
            f"Exported DXF (RMSE {result.report.get('rmse', '?')} m, "
            f"threshold {result.report.get('threshold_status', '?')})."
        )
        messagebox.showinfo(
            "Export complete",
            "\n".join([
                f"DXF:   {result.dxf_path}",
                f"JSON:  {result.json_path}",
                f"HTML:  {result.html_path}",
            ]),
            parent=self,
        )

    def _on_export_failure(self, exc: BaseException) -> None:
        self._btn_export.configure(state=tk.NORMAL)
        messagebox.showerror("Export failed", str(exc), parent=self)
        self._set_status(f"Export failed: {exc}")


class _TemporaryGcpFile:
    """Context manager that stages the GUI's GCPs for the pipeline."""

    def __init__(self, state: GeorefGuiState) -> None:
        self._state = state
        self._path: Optional[Path] = None

    def __enter__(self) -> Path:
        import tempfile

        fd, name = tempfile.mkstemp(prefix="pdf2geocad-gui-", suffix=".gcps.json")
        import os
        os.close(fd)
        self._path = Path(name)
        save_gcps(self._path, self._state.gcps)
        return self._path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._path is not None:
            self._path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Tk entry point used by the ``pdf2geocad-gui`` script and ``--gui``."""

    if argv is None:
        import sys

        argv = sys.argv[1:]
    arguments = list(argv)

    if arguments == ["--self-test"]:
        from librecad_pdf_importer.runtime_self_test import run_runtime_self_test

        return run_runtime_self_test()

    pdf_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    if arguments:
        positional = [item for item in arguments if not str(item).startswith("-")]
        if positional:
            pdf_path = Path(positional[0])
        for index, item in enumerate(arguments):
            if item in {"--output-dir", "-o"} and index + 1 < len(arguments):
                output_dir = Path(arguments[index + 1])

    app = GeorefGuiApp(pdf_path=pdf_path, output_dir=output_dir)
    try:
        app.mainloop()
    finally:
        try:
            app.destroy()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
