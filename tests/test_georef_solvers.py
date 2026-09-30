from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math

import pytest

from librecad_pdf_importer.georef import (
    GCP,
    GeoreferenceError,
    load_gcps,
    save_gcps,
    solve_affine,
    solve_helmert,
    validate_crs,
)


def test_gcp_json_round_trip_uses_versioned_schema(tmp_path) -> None:
    gcps = (
        GCP(10.5, -2.0, 500_000.0, 4_100_000.0, label="A"),
        GCP(30.0, 8.25, 500_020.0, 4_100_010.25, label="B"),
    )
    path = tmp_path / "drawing.gcps.json"

    save_gcps(path, gcps)

    assert load_gcps(path) == gcps
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["gcps"][0] == {
        "label": "A",
        "source_x": 10.5,
        "source_y": -2.0,
        "world_x": 500_000.0,
        "world_y": 4_100_000.0,
    }


def test_gcp_is_immutable_and_json_schema_errors_are_domain_errors(tmp_path) -> None:
    gcp = GCP(0.0, 0.0, 1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        gcp.source_x = 3.0

    path = tmp_path / "bad.gcps.json"
    path.write_text('{"schema_version": 1, "gcps": [{"source_x": 0}]}', encoding="utf-8")
    with pytest.raises(GeoreferenceError, match="GCP 1"):
        load_gcps(path)


@pytest.mark.parametrize("schema_version", [True, False, 1.0, "1", None])
def test_load_gcps_rejects_boolean_or_wrong_type_schema_version(
    tmp_path, schema_version
) -> None:
    path = tmp_path / "bad-version.gcps.json"
    path.write_text(
        json.dumps({"schema_version": schema_version, "gcps": []}),
        encoding="utf-8",
    )

    with pytest.raises(GeoreferenceError, match="schema_version"):
        load_gcps(path)


def test_load_gcps_wraps_oversized_json_integer_parser_error(tmp_path) -> None:
    path = tmp_path / "oversized-parser-number.gcps.json"
    path.write_text(
        '{"schema_version":1,"gcps":[{"source_x":'
        + "9" * 5_000
        + ',"source_y":0,"world_x":0,"world_y":0}]}',
        encoding="utf-8",
    )

    with pytest.raises(GeoreferenceError):
        load_gcps(path)


def test_load_gcps_wraps_coordinate_float_overflow(tmp_path) -> None:
    path = tmp_path / "oversized-coordinate.gcps.json"
    path.write_text(
        '{"schema_version":1,"gcps":[{"source_x":'
        + "9" * 400
        + ',"source_y":0,"world_x":0,"world_y":0}]}',
        encoding="utf-8",
    )

    with pytest.raises(GeoreferenceError, match="source_x.*finite"):
        load_gcps(path)


def test_validate_crs_accepts_local_and_canonicalizes_epsg() -> None:
    local = validate_crs("local")
    assert local.mode == "local"
    assert local.authority is None

    projected = validate_crs("32652")
    assert projected.mode == "epsg"
    assert projected.authority == "EPSG:32652"
    assert projected.name


@pytest.mark.parametrize(
    "value",
    ["EPSG:999999", "OGC:CRS84"],
)
def test_validate_crs_rejects_invalid_or_non_epsg_input(value) -> None:
    with pytest.raises(GeoreferenceError, match="EPSG"):
        validate_crs(value)


@pytest.mark.parametrize("value", ["EPSG:4326", "EPSG:2277", "EPSG:4978"])
def test_validate_crs_requires_projected_2d_metre_epsg(value) -> None:
    with pytest.raises(GeoreferenceError, match="projected 2D.*metre"):
        validate_crs(value)


def test_helmert_least_squares_recovers_coefficients_and_qc() -> None:
    scale = 2.0
    rotation_deg = 30.0
    a = scale * math.cos(math.radians(rotation_deg))
    b = scale * math.sin(math.radians(rotation_deg))

    def world(x, y):
        return a * x - b * y + 100.0, b * x + a * y - 50.0

    gcps = tuple(
        GCP(x, y, *world(x, y), label=str(index))
        for index, (x, y) in enumerate(((0, 0), (10, 0), (0, 5), (4, 7)), start=1)
    )

    result = solve_helmert(gcps)

    assert result.method == "helmert"
    assert result.transform_coefficients.a == pytest.approx(a)
    assert result.transform_coefficients.b == pytest.approx(b)
    assert result.transform_coefficients.tx == pytest.approx(100.0)
    assert result.transform_coefficients.ty == pytest.approx(-50.0)
    assert result.scale == pytest.approx(scale)
    assert result.rotation_deg == pytest.approx(rotation_deg)
    assert result.transform_point(3.0, 2.0) == pytest.approx(world(3.0, 2.0))
    assert result.rmse == pytest.approx(0.0, abs=1e-12)
    assert result.max_residual == pytest.approx(0.0, abs=1e-12)
    assert len(result.residuals) == len(gcps)


def test_helmert_reports_signed_residuals_and_radial_statistics() -> None:
    gcps = (
        GCP(0, 0, 5.0, -3.0),
        GCP(10, 0, 15.0, -3.0),
        GCP(0, 10, 5.0, 7.0),
        GCP(10, 10, 15.4, 6.8),
    )

    result = solve_helmert(gcps)

    errors = []
    for gcp, residual in zip(gcps, result.residuals, strict=True):
        predicted_x, predicted_y = result.transform_point(*gcp.source)
        assert residual.gcp == gcp
        assert residual.residual_x == pytest.approx(predicted_x - gcp.world_x)
        assert residual.residual_y == pytest.approx(predicted_y - gcp.world_y)
        assert residual.total_error == pytest.approx(
            math.hypot(residual.residual_x, residual.residual_y)
        )
        errors.append(residual.total_error)
    assert result.rmse == pytest.approx(math.sqrt(sum(error**2 for error in errors) / 4))
    assert result.max_residual == pytest.approx(max(errors))


def test_affine_least_squares_recovers_all_six_coefficients() -> None:
    expected = (1.5, 0.25, 100.0, -0.4, 0.8, 200.0)

    def world(x, y):
        a, b, c, d, e, f = expected
        return a * x + b * y + c, d * x + e * y + f

    gcps = tuple(
        GCP(x, y, *world(x, y))
        for x, y in ((0, 0), (10, 0), (0, 5), (8, 9))
    )

    result = solve_affine(gcps)

    coefficients = result.transform_coefficients
    assert result.method == "affine"
    assert (
        coefficients.a,
        coefficients.b,
        coefficients.c,
        coefficients.d,
        coefficients.e,
        coefficients.f,
    ) == pytest.approx(expected)
    assert result.transform_point(2.5, -4.0) == pytest.approx(world(2.5, -4.0))
    assert result.scale is None
    assert result.rotation_deg is None
    assert result.rmse == pytest.approx(0.0, abs=1e-11)


@pytest.mark.parametrize(
    ("solver", "gcps", "message"),
    [
        (solve_helmert, (GCP(0, 0, 1, 1),), "at least 2"),
        (
            solve_helmert,
            (GCP(0, 0, 1, 1), GCP(0, 0, 2, 2)),
            "rank-deficient",
        ),
        (
            solve_affine,
            (GCP(0, 0, 1, 1), GCP(1, 0, 2, 1)),
            "at least 3",
        ),
        (
            solve_affine,
            (GCP(0, 0, 1, 1), GCP(1, 1, 2, 2), GCP(2, 2, 3, 3)),
            "rank-deficient",
        ),
    ],
)
def test_solvers_reject_insufficient_or_rank_deficient_gcps(solver, gcps, message) -> None:
    with pytest.raises(GeoreferenceError, match=message):
        solver(gcps)


@pytest.mark.parametrize(
    ("solver", "gcps"),
    [
        (
            solve_helmert,
            (
                GCP(0, 0, 10, 20),
                GCP(0, 0, 10, 20),
                GCP(5, 0, 15, 20),
            ),
        ),
        (
            solve_affine,
            (
                GCP(0, 0, 10, 20),
                GCP(0, 0, 10, 20),
                GCP(5, 0, 15, 20),
                GCP(0, 5, 10, 25),
            ),
        ),
    ],
)
def test_solvers_reject_duplicate_source_gcps_even_when_layout_has_full_rank(solver, gcps) -> None:
    with pytest.raises(GeoreferenceError, match="duplicate"):
        solver(gcps)


@pytest.mark.parametrize(
    ("solver", "gcps"),
    [
        (
            solve_helmert,
            (GCP(0, 0, 5, 5), GCP(1, 0, 5, 5)),
        ),
        (
            solve_affine,
            (GCP(0, 0, 5, 5), GCP(1, 0, 5, 5), GCP(0, 1, 5, 5)),
        ),
    ],
)
def test_solvers_reject_collapsed_destination_layout(solver, gcps) -> None:
    with pytest.raises(GeoreferenceError, match="destination"):
        solver(gcps)


@pytest.mark.parametrize(
    ("solver", "gcps"),
    [
        (
            solve_helmert,
            (GCP(0, 0, 1e308, 1e308), GCP(1, 0, -1e308, -1e308)),
        ),
        (
            solve_affine,
            (
                GCP(0, 0, 1e308, 1e308),
                GCP(1, 0, -1e308, -1e308),
                GCP(0, 1, 1e308, -1e308),
            ),
        ),
    ],
)
def test_solvers_reject_non_finite_solved_mappings(solver, gcps) -> None:
    with pytest.raises(GeoreferenceError, match="finite"):
        solver(gcps)


def test_affine_rejects_singular_destination_mapping() -> None:
    gcps = (
        GCP(0, 0, 0, 0),
        GCP(1, 0, 1, 0),
        GCP(0, 1, 2, 0),
    )

    with pytest.raises(GeoreferenceError, match="destination|invertible"):
        solve_affine(gcps)
