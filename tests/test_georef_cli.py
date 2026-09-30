"""Public command-line surface for PDF2GeoCAD."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

import pymupdf as fitz

from librecad_pdf_importer.georef import cli
from librecad_pdf_importer.georef.gcp import save_gcps
from librecad_pdf_importer.georef.models import GCP


ROOT = Path(__file__).resolve().parents[1]
MM_PER_PT = 25.4 / 72.0


def _write_cli_inputs(tmp_path: Path, *, valid: bool = True) -> tuple[Path, Path]:
    source = tmp_path / "drawing.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.draw_line((20, 30), (100, 80), color=(0, 0, 0), width=1)
    document.save(source)
    document.close()
    source_points = (
        (20 * MM_PER_PT, (100 - 30) * MM_PER_PT),
        (100 * MM_PER_PT, (100 - 80) * MM_PER_PT),
    )
    if not valid:
        source_points = source_points[:1]
    gcps = [
        GCP(x, y, 2.0 * x + 1000.0, 2.0 * y + 2000.0, f"P{index}")
        for index, (x, y) in enumerate(source_points, start=1)
    ]
    gcp_path = tmp_path / "drawing.gcps.json"
    save_gcps(gcp_path, gcps)
    return source, gcp_path


def test_parser_exposes_approved_conversion_options():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "drawing.pdf",
            "--gcp",
            "drawing.gcps.json",
            "--transform",
            "affine",
            "--crs",
            "EPSG:5186",
            "--output-dir",
            "out",
            "--rmse-threshold",
            "0.25",
            "--dxf-version",
            "R2013",
            "--no-text",
            "--text-mode",
            "geometry",
            "--verbose",
        ]
    )

    assert args.input == "drawing.pdf"
    assert args.gcp == "drawing.gcps.json"
    assert args.transform == "affine"
    assert args.crs == "EPSG:5186"
    assert args.output_dir == "out"
    assert args.rmse_threshold == 0.25
    assert args.dxf_version == "R2013"
    assert args.include_text is False
    assert args.text_mode == "geometry"
    assert args.verbose is True


def test_parser_does_not_advertise_r12_for_georeferenced_output():
    action = next(
        action for action in cli.build_parser()._actions if action.dest == "dxf_version"
    )
    assert tuple(action.choices) == (
        "R2000",
        "R2004",
        "R2007",
        "R2010",
        "R2013",
        "R2018",
    )


def test_module_help_lists_public_options():
    completed = subprocess.run(
        [sys.executable, "-m", "librecad_pdf_importer.georef.cli", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for option in (
        "--gcp",
        "--transform",
        "--crs",
        "--output-dir",
        "--rmse-threshold",
        "--dxf-version",
        "--no-text",
        "--text-mode",
        "--gui",
        "--verbose",
    ):
        assert option in completed.stdout


def test_gui_flag_uses_late_launch_hook_without_conversion(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_launch_gui", lambda input_path=None: calls.append(input_path) or 0)
    monkeypatch.setattr(
        cli,
        "run_georef_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("conversion ran")),
    )

    assert cli.main(["--gui"]) == 0
    assert calls == [None]


def test_self_test_uses_runtime_probe_without_conversion(monkeypatch):
    from librecad_pdf_importer import runtime_self_test

    calls = []
    monkeypatch.setattr(
        runtime_self_test,
        "run_runtime_self_test",
        lambda: calls.append("self-test") or 0,
    )
    monkeypatch.setattr(
        cli,
        "run_georef_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("conversion ran")),
    )

    assert cli.main(["--self-test"]) == 0
    assert calls == ["self-test"]


def test_gui_launch_preserves_the_complete_input_path(monkeypatch):
    from librecad_pdf_importer.georef import gui

    launched = {}

    class FakeApp:
        def __init__(self, *, pdf_path=None, output_dir=None):
            launched["pdf_path"] = pdf_path
            launched["output_dir"] = output_dir

        def mainloop(self):
            return None

        def destroy(self):
            return None

    monkeypatch.setattr(gui, "GeorefGuiApp", FakeApp)

    assert cli._launch_gui("drawing.pdf") == 0
    assert launched == {"pdf_path": Path("drawing.pdf"), "output_dir": None}


def test_cli_success_prints_qc_and_output_summary(tmp_path):
    source, gcp_path = _write_cli_inputs(tmp_path)
    output_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "librecad_pdf_importer.georef.cli",
            str(source),
            "--gcp",
            str(gcp_path),
            "--output-dir",
            str(output_dir),
            "--rmse-threshold",
            "0.01",
            "--no-text",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert completed.stderr == ""
    for label in ("STATUS:", "TRANSFORM:", "CRS:", "RMSE:", "DXF:", "JSON:", "HTML:"):
        assert label in completed.stdout
    assert {path.name for path in output_dir.iterdir()} == {
        "drawing_georef.dxf",
        "drawing_georef.json",
        "drawing_georef_report.html",
    }


def test_cli_invalid_gcps_return_domain_error_without_artifacts(tmp_path):
    source, gcp_path = _write_cli_inputs(tmp_path, valid=False)
    output_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "librecad_pdf_importer.georef.cli",
            str(source),
            "--gcp",
            str(gcp_path),
            "--output-dir",
            str(output_dir),
            "--no-text",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("ERROR: ")
    assert "at least 2 GCPs" in completed.stderr
    assert not output_dir.exists()


def test_cli_rejects_r12_with_exit_2_and_no_artifacts(tmp_path):
    source, gcp_path = _write_cli_inputs(tmp_path)
    output_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "librecad_pdf_importer.georef.cli",
            str(source),
            "--gcp",
            str(gcp_path),
            "--output-dir",
            str(output_dir),
            "--dxf-version",
            "R12",
            "--no-text",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "invalid choice" in completed.stderr
    assert "R12" in completed.stderr
    assert not output_dir.exists()


def test_pyproject_registers_pdf2geocad_console_script():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["pdf2geocad"] == (
        "librecad_pdf_importer.georef.cli:main"
    )
