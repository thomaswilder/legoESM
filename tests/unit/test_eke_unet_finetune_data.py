"""Tests for the 1 deg EKE U-Net fine-tuning data and scoring scripts.

``scripts/data/prepare_eke_unet_finetune_data.py`` (split and gap rule, scored-cell
rule, the |f| floor on a row with f = 0, the by-name transpose and the read-back
checks, on small synthetic NetCDF files) and ``scripts/validate/eke_unet_1deg_skill.py``
(the section 6 scores on cases with a known answer). The scripts are loaded from
their paths: they are not a package.
"""

import importlib.util
import pathlib

import jax
import numpy as np
import pytest
import xarray as xr
import yaml

jax.config.update("jax_enable_x64", True)

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts" / "experiment" / "dino" / "eke_unet_finetune_1deg.yaml"


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load("scripts/data/prepare_eke_unet_finetune_data.py", "prepare_eke_unet_finetune_data")
skill = _load("scripts/validate/eke_unet_1deg_skill.py", "eke_unet_1deg_skill")


# ---- split and gap rule ----

def test_config_split_has_the_spec_sizes_and_gaps():
    cfg = yaml.safe_load(open(CONFIG))
    idx = prep.split_indices(cfg["splits"], cfg["data"]["n_days"])
    assert [idx[k].size for k in ("train", "validation", "test")] == [3150, 270, 720]
    assert idx["validation"][0] - idx["train"][-1] == 91 and idx["test"][0] - idx["validation"][-1] == 91


@pytest.mark.parametrize("splits", (
    {"train": [0, 3149], "validation": [3239, 3509], "test": [3600, 4319]},    # 90-day gap, one day short
    {"train": [0, 3149], "validation": [3240, 3509], "test": [3599, 4319]},    # same, validation to test
    {"train": [0, 3149], "validation": [3100, 3509], "test": [3600, 4319]},    # overlap
    {"train": [0, 3149], "validation": [3240, 3509], "test": [3600, 4320]},    # past the record
))
def test_split_refuses_short_gaps_overlap_and_days_outside_the_record(splits):
    with pytest.raises(ValueError):
        prep.split_indices({**splits, "min_separation_days": 91}, 4320)


# ---- scored cells ----

def test_scored_cells_rule():
    tmask = np.array([[0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]], bool)
    wc = np.array([[np.nan, 1.0, 0.49, 0.5, 1.0],
                   [0.9, 1.0, 1.0, 1.0, 0.3]])
    got = prep.scored_cells(tmask, wc, dropped_columns=[0, 4], coverage_min=0.5)
    # land (0, 0); coverage 0.49 (0, 2) out and 0.5 (0, 3) in; columns 0 and 4 out
    assert got.tolist() == [[False, True, False, True, False],
                            [False, True, True, True, False]]
    assert tmask.sum() == 9  # the input is not modified
    assert prep.scored_cells(tmask, wc, [], 0.5).sum() == 7


# ---- |f| floor ----

def test_floored_f_floors_magnitude_keeps_sign_and_takes_plus_at_zero():
    ff = np.array([-1e-4, -1e-9, 0.0, 1e-9, 1e-4])
    assert prep.floored_f(ff, 1e-7).tolist() == [-1e-4, -1e-7, 1e-7, 1e-7, 1e-4]
    # float32 f must not round the floor to float32
    assert prep.floored_f(np.zeros(2, np.float32), 1e-7).tolist() == [1e-7, 1e-7]


def test_floored_f_is_the_denominator_of_surface_vorticity_over_f():
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ml.eke_unet import F_MIN, surface_vorticity_over_f
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre

    grid = create_latlon_grid(n_lat=15, n_lon=8, radius=constants.R_earth)  # middle row on the equator
    rng = np.random.default_rng(1)
    u = jnp.asarray(0.1 * rng.standard_normal((15, 9)))
    v = jnp.asarray(0.1 * rng.standard_normal((16, 8)))
    f = np.asarray(grid.grid_coriolis)
    assert (np.abs(f[7]) < F_MIN).all()
    zeta = np.asarray(relative_vorticity_cell_centre(u, v, grid))
    np.testing.assert_allclose(zeta / prep.floored_f(f, F_MIN), np.asarray(surface_vorticity_over_f(u, v, grid)), rtol=1e-12)


def test_weighted_mean_std_is_two_pass_in_float64():
    # float32 with a large offset: the one-pass <x^2> - <x>^2 would lose the variance entirely
    rng = np.random.default_rng(0)
    x = (1.0e4 + rng.standard_normal((200, 300)) * 1e-2).astype(np.float32)
    w = rng.uniform(1, 2, (300,))
    m, s = prep.weighted_mean_std(x, w)
    x64 = x.astype(np.float64)
    ref_m = np.average(x64, weights=np.broadcast_to(w, x.shape))
    ref_s = np.sqrt(np.average((x64 - ref_m) ** 2, weights=np.broadcast_to(w, x.shape)))
    assert m == pytest.approx(ref_m, rel=1e-14) and s == pytest.approx(ref_s, rel=1e-12)
    assert 0.9e-2 < s < 1.1e-2


# ---- reading the input files ----

NY, NX, NT = 7, 5, 6
COLS = [0, 4]


def _write_inputs(tmp_path, vor_dims=("y_c", "x_c", "t"), mutate=None):
    rng = np.random.default_rng(0)
    tmask = np.zeros((NY, NX), bool)
    tmask[1:6, 1:4] = True       # 15 interior wet cells
    tmask[2:5, COLS] = True      # ring columns wet on rows 2-4 only
    ff = np.full((NY, NX), 1.0e-4)
    ff[3] = 0.0
    wc = np.where(tmask, 1.0, np.nan)
    wc[2:5, COLS] = 0.3          # 6 low-coverage wet cells, all in the ring columns
    vor = np.where(tmask[:, :, None], rng.standard_normal((NY, NX, NT)), 0.0)
    eke = np.where(tmask[None], rng.uniform(0.1, 1.0, (NT, NY, NX)), 0.0)
    if mutate:
        vor, eke = mutate(vor, eke)
    t = np.arange(NT, dtype=float)
    xr.Dataset({"vor": (("y_c", "x_c", "t"), vor)}, coords={"t": t}) \
        .transpose(*vor_dims).to_netcdf(tmp_path / "vor.nc")
    xr.Dataset({"fine_ke": (("t", "y_c", "x_c"), eke)}, coords={"t": t}).to_netcdf(tmp_path / "eke.nc")
    xr.Dataset({"tmask": (("t", "nav_lev", "y_c", "x_c"), tmask[None, None].astype(float)),
                **{k: (("t", "y_c", "x_c"), a[None]) for k, a in
                   (("ff_t", ff), ("e1t", np.full((NY, NX), 2.0)), ("e2t", np.full((NY, NX), 3.0)),
                    ("gphit", np.linspace(-60, 60, NY)[:, None] * np.ones((NY, NX))),
                    ("glamt", np.ones((NY, 1)) * np.arange(NX)[None]))}}).to_netcdf(tmp_path / "mesh.nc")
    xr.Dataset({"wet_coverage": (("y_c", "x_c"), wc)}).to_netcdf(tmp_path / "cov.nc")
    return vor, eke, {
        "dir": str(tmp_path), "vor_file": "vor.nc", "eke_file": "eke.nc", "mesh_file": "mesh.nc", "coverage_file": "cov.nc",
        "vor_var": "vor", "eke_var": "fine_ke", "coverage_var": "wet_coverage", "n_days": NT,
        "expect": {"ny": NY, "nx": NX, "wet_cells": 21, "scored_cells": 15, "low_coverage_wet_cells": 6,
                   "zero_f_rows": [3], "land_rows": [0, 6], "ring_wet_rows": [2, 4]}}


def test_read_inputs_transposes_vor_by_name_and_reads_back(tmp_path, capsys):
    vor, eke, data = _write_inputs(tmp_path)
    r = prep.read_inputs(data, COLS, 0.5)
    assert r["vor"].shape == (NT, NY, NX) and r["eke"].shape == (NT, NY, NX)   # the sizes all differ
    np.testing.assert_array_equal(r["vor"], np.moveaxis(vor, 2, 0))
    np.testing.assert_array_equal(r["eke"], eke)
    assert r["scored"].sum() == 15 and r["tmask"].sum() == 21 and not r["scored"][:, COLS].any()
    assert r["area"].max() == 6.0 and r["ff"][3].max() == 0.0
    assert "PASS  1 scored cells" in capsys.readouterr().out


def test_read_inputs_refuses_vor_with_another_dimension_order(tmp_path):
    _, _, data = _write_inputs(tmp_path, vor_dims=("t", "y_c", "x_c"))
    with pytest.raises(SystemExit, match="1 dims"):
        prep.read_inputs(data, COLS, 0.5)


@pytest.mark.parametrize("what, mutate, match", (
    ("a NaN in vor", lambda v, e: (np.where(np.arange(NT) == 2, np.nan, v), e), "1 NaN"),
    ("a zero EKE on a wet cell", lambda v, e: (v, np.where((np.arange(NT)[:, None, None] == 1) & (np.arange(NY)[None, :, None] == 2)
                                                           & (np.arange(NX)[None, None, :] == 2), 0.0, e)), "1 EKE > 0"),
    ("vorticity on land", lambda v, e: (v + 1e-3, e), "1 land is 0"),
))
def test_read_inputs_stops_on_a_failed_read_back(tmp_path, what, mutate, match):
    _, _, data = _write_inputs(tmp_path, mutate=mutate)
    with pytest.raises(SystemExit, match=match):
        prep.read_inputs(data, COLS, 0.5)


def test_read_inputs_stops_when_a_low_coverage_cell_is_outside_the_dropped_columns(tmp_path):
    _, _, data = _write_inputs(tmp_path)
    with pytest.raises(SystemExit, match="1 low-coverage cells"):
        prep.read_inputs(data, [0], 0.5)  # column 4 is no longer dropped: its 3 low-coverage cells are outside the dropped columns


# ---- scores ----

def _case(n=40, m=30, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(-3.0, 1.5, (n, m))
    return y, rng.uniform(1.0, 5.0, m), rng.normal(-3.0, 1.0, m)


def test_score_perfect_shifted_and_constant_predictions():
    y, area, c = _case()
    s = skill.score(y, y, area, c, 10.0)
    assert (s["r2"], s["bias"], s["mean_ratio"], s["mean_ratio_area"], s["low10"]) == (1.0, 0.0, 1.0, 1.0, 0.0)
    assert s["anomaly_corr"] == pytest.approx(1.0) and s["corr_log"] == pytest.approx(1.0) and s["n"] == y.size
    s = skill.score(y, y + np.log(2.0), area, c, 10.0)
    assert s["bias"] == pytest.approx(np.log(2.0)) and s["exp_bias"] == pytest.approx(2.0) and s["mean_ratio"] == pytest.approx(2.0)
    s = skill.score(y, np.full_like(y, y.mean()), area, c, 10.0)
    assert s["r2"] == pytest.approx(0.0, abs=1e-12)


def test_score_low10_is_a_strict_factor_of_ten_below():
    y, area, c = _case()
    assert skill.score(y, y - np.log(20.0), area, c, 10.0)["low10"] == 1.0
    assert skill.score(y, y - np.log(5.0), area, c, 10.0)["low10"] == 0.0


def test_score_area_weighted_ratio_and_anomaly_correlation():
    y = np.log(np.array([[1.0, 3.0]]))
    p = np.log(np.array([[2.0, 3.0]]))
    s = skill.score(y, p, np.array([3.0, 1.0]), None, 10.0)
    assert s["mean_ratio"] == pytest.approx(5.0 / 4.0) and s["mean_ratio_area"] == pytest.approx(9.0 / 6.0)
    assert s["anomaly_corr"] is None
    yy, area, c = _case()
    assert skill.score(yy, c + 2.0 * (yy - c), area, c, 10.0)["anomaly_corr"] == pytest.approx(1.0)
    assert skill.score(yy, c + 2.0 * (c - yy), area, c, 10.0)["anomaly_corr"] == pytest.approx(-1.0)


def test_score_accumulates_float32_inputs_in_float64():
    rng = np.random.default_rng(2)
    y = rng.normal(-3.0, 1.5, (600, 2000)).astype(np.float32)
    p = (y + rng.normal(0.0, 0.3, y.shape)).astype(np.float32)
    area, c = rng.uniform(1, 2, 2000), rng.normal(-3, 1, 2000)
    a = skill.score(y, p, area, c, 10.0)
    b = skill.score(y.astype(np.float64), p.astype(np.float64), area, c, 10.0)
    assert a == b


def test_band_and_box_masks_are_half_open_and_closed():
    lat = np.array([0.0, 4.999, 5.0, 19.999, 20.0, 60.0])
    assert skill.band_mask(lat, 5.0, 20.0).tolist() == [False, False, True, True, False, False]
    box = {"lon": [0.0, 10.0], "lat": [-1.0, 1.0]}
    assert skill.box_mask(np.array([0.0, 10.0, 10.1]), np.array([-1.0, 1.0, 0.0]), box).tolist() == [True, True, False]
