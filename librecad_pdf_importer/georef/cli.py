"""Command-line entry point for metre-based manual-GCP PDF georeferencing."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from .models import GeoreferenceError
from .pipeline import GEOREF_DXF_VERSIONS, run_georef_pipeline


DXF_VERSIONS = GEOREF_DXF_VERSIONS
TEXT_MODES = ("text", "labels", "3d_text", "glyphs", "geometry")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2geocad",
        description="Convert page 1 of a vector PDF to a real-coordinate metre DXF.",
    )
    parser.add_argument("input", nargs="?", help="input vector PDF")
    parser.add_argument("--gcp", help="Version 1 GCP JSON file")
    parser.add_argument(
        "--transform",
        choices=("helmert", "affine"),
        default="helmert",
        help="georeferencing transform (default: helmert)",
    )
    parser.add_argument(
        "--crs",
        default="local",
        help="local or a projected metre EPSG CRS (default: local)",
    )
    parser.add_argument("--output-dir", help="artifact directory (default: input directory)")
    parser.add_argument(
        "--rmse-threshold",
        type=float,
        default=1.0,
        help="warning threshold in output metres (default: 1.0)",
    )
    parser.add_argument(
        "--dxf-version",
        choices=DXF_VERSIONS,
        default="R2018",
        help="DXF version (default: R2018)",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--include-text",
        "--import-text",
        "--text",
        dest="include_text",
        action="store_true",
        help="include extracted text (default)",
    )
    text_group.add_argument(
        "--no-text",
        "--no-import-text",
        "--no-include-text",
        dest="include_text",
        action="store_false",
        help="exclude text",
    )
    parser.set_defaults(include_text=True)
    parser.add_argument(
        "--text-mode",
        choices=TEXT_MODES,
        default="text",
        help="text representation (default: text)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch the GUI through a late import",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify bundled runtime dependencies and exit",
    )
    parser.add_argument("--verbose", action="store_true", help="show traceback on failure")
    return parser


def _print_line(text: str, *, file=None) -> None:
    target = file or sys.stdout
    try:
        print(text, file=target)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"), file=target)


def _launch_gui(input_path: str | None = None) -> int:
    from .gui import main as gui_main

    return int(gui_main([input_path] if input_path else []) or 0)


def _print_success(result) -> None:
    report = result.report
    _print_line(f"STATUS: {report['status']} ({report['threshold_status']})")
    _print_line(f"TRANSFORM: {report['method']}")
    _print_line(f"CRS: {report['crs']['label']}")
    _print_line(f"RMSE: {report['rmse']:.9g} m")
    _print_line(f"MAX_RESIDUAL: {report['max_residual']:.9g} m")
    _print_line(f"DXF: {result.dxf_path}")
    _print_line(f"JSON: {result.json_path}")
    _print_line(f"HTML: {result.html_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        from librecad_pdf_importer.runtime_self_test import run_runtime_self_test

        return run_runtime_self_test()
    if args.gui:
        try:
            return _launch_gui(args.input)
        except (ImportError, ModuleNotFoundError) as exc:
            _print_line(f"ERROR: GUI unavailable: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            if args.verbose:
                traceback.print_exc()
            _print_line(f"ERROR: {exc}", file=sys.stderr)
            return 3

    if not args.input:
        _print_line("ERROR: input PDF is required", file=sys.stderr)
        return 1
    if not args.gcp:
        _print_line("ERROR: --gcp is required", file=sys.stderr)
        return 1

    try:
        result = run_georef_pipeline(
            Path(args.input),
            Path(args.gcp),
            transform=args.transform,
            crs=args.crs,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            rmse_threshold=args.rmse_threshold,
            dxf_version=args.dxf_version,
            include_text=args.include_text,
            text_mode=args.text_mode,
        )
    except GeoreferenceError as exc:
        _print_line(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        _print_line(f"ERROR: {exc}", file=sys.stderr)
        return 3

    _print_success(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
