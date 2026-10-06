"""Exercise the distributed CLI without the source tree on its import path."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def cli_wheel(tmp_path_factory):
    pytest.importorskip("setuptools.build_meta", reason="requires the declared wheel build backend")
    root = Path(__file__).resolve().parents[1]
    project = tmp_path_factory.mktemp("wheel_source")
    for path in [root / "pyproject.toml", root / "README.md", root / "LICENSE", *root.glob("*.py")]:
        shutil.copy2(path, project / path.name)
    for package in ("pdfcadcore", "librecad_pdf_importer"):
        shutil.copytree(
            root / package, project / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    output = tmp_path_factory.mktemp("wheel_dist")
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])",
         str(output)],
        cwd=project, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next(output.glob("*.whl"))


@pytest.mark.parametrize("module", ["pdf2dxf", "librecad_pdf_importer.cli"])
def test_wheel_cli_rejects_non_pdf_cleanly(cli_wheel, tmp_path, module):
    pdf = tmp_path / "not-a-pdf.pdf"
    pdf.write_text("not a PDF", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import importlib, sys; sys.path.insert(0, sys.argv.pop(1)); "
         "entry = importlib.import_module(sys.argv.pop(1)); raise SystemExit(entry.main())",
         str(cli_wheel), module, str(pdf)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "not a valid PDF" in result.stderr
    assert "Traceback" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
