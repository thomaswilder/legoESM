"""Direct test of ``scripts/data/coarse_grain_dino_state.py``.

Synthetic pair built analytically by ``nemo_dino_mesh``: DINO 1/4 deg as the
"fine" mesh, DINO 1 deg as the "coarse" one.  Same structure as the real
1/16 -> 1/4 pair: one land ring column per side (so the coarse ring columns are
only partly covered), periodic channel, rows that do not nest, an equator row,
and deep coarse cells with no wet fine cell under them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import netCDF4  # noqa: E402
from legoesm.ocean.fidelity.nemo_dino_mesh import (  # noqa: E402
    NEMO_DINO_R1,
    NEMO_DINO_R4,
    nemo_dino_hgr,
    nemo_dino_mesh,
)

from scripts.data import coarse_grain_dino_state as cg  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    meshes = {}
    for key, nml in (("fine", NEMO_DINO_R4), ("coarse", NEMO_DINO_R1)):
        g, h = nemo_dino_mesh(nml), nemo_dino_hgr(nml)
        meshes[key] = dict(
            glamf=h["glamf"], gphif=h["gphif"], gphit=h["gphit"],
            tmask=np.moveaxis(g.tmask, -1, 0).astype(bool),       # (z, y, x)
            e3t_0=g.e3t_0[0, 0].copy(), e3t_1d=g.e3t_1d, gdept_1d=g.gdept_1d,
            nav_lat=h["gphit"].astype(np.float32), nav_lon=h["glamt"].astype(np.float32),
        )
    f, c = meshes["fine"], meshes["coarse"]
    w = cg.overlap_weights(f["glamf"], f["gphif"], c["glamf"], c["gphif"])
    cov = cg.regrid(f["tmask"], w)
    return f, c, w, cov


def test_constant_field_kept_and_dry_cells_zero(pair):
    f, c, w, cov = pair
    need = c["tmask"] & (cov == 0.0)
    assert need.sum() > 0          # the fill-from-above path is exercised
    assert np.any(cov[:, :, 0][c["tmask"][:, :, 0]] < 0.5)   # ring column partly covered
    field = np.where(f["tmask"], 7.25, 123.0)   # land value must not leak
    out = cg.coarse_grain(field, f["tmask"], c["tmask"], cov, w)
    assert np.max(np.abs(out[c["tmask"]] - 7.25)) < 1e-12
    assert np.all(out[~c["tmask"]] == 0.0)


def test_content_closes_with_lost_part(pair):
    f, c, w, cov = pair
    rng = np.random.default_rng(0)
    field = rng.normal(10.0, 3.0, f["tmask"].shape)
    out = cg.coarse_grain(field, f["tmask"], c["tmask"], cov, w)
    chk = cg.content_check(field, out, f["tmask"], c["tmask"], cov, w,
                           cg.cell_areas(f["glamf"], f["gphif"]),
                           cg.cell_areas(c["glamf"], c["gphif"]), f["e3t_0"])
    assert chk["lost"] != 0.0      # some fine water lies under dry coarse cells
    assert chk["rel_closure"] < 1e-12
    assert chk["rel_mean_diff"] < 1e-12
    assert chk["n_nonfinite"] == 0 and chk["max_abs_dry"] == 0.0


def _fine_rows_height_bound(f_edges, c_edges):
    """Largest fine row height (deg) overlapping each coarse row."""
    lo, hi = f_edges[:-1], f_edges[1:]
    return np.array([np.max((hi - lo)[(hi > a) & (lo < b)])
                     for a, b in zip(c_edges[:-1], c_edges[1:])])


def test_latitude_registration(pair):
    f, c, w, _ = pair
    ones_f = np.ones((1,) + f["gphit"].shape, dtype=bool)
    ones_c = np.ones((1,) + c["gphit"].shape, dtype=bool)
    cov1 = cg.regrid(ones_f, w)
    got = cg.coarse_grain(f["gphit"][None], ones_f, ones_c, cov1, w)[0, :, 10]
    # exact area-mean latitude of each coarse row: int(phi cos) / int(cos)
    ce = cg.cell_edges_rad(c["gphif"][:, 0])
    ref = np.rad2deg(np.diff(ce * np.sin(ce) + np.cos(ce)) / np.diff(np.sin(ce)))
    tol = _fine_rows_height_bound(np.rad2deg(cg.cell_edges_rad(f["gphif"][:, 0])),
                                  np.rad2deg(ce))
    inner = slice(1, -1)           # rows 0 and n-1 are only partly covered
    assert np.all(np.abs(got - ref)[inner] <= tol[inner])
    # a one-row shift of the coarse latitudes must fail the same bound
    assert np.all(np.abs(got[1:-1] - ref[:-2]) > tol[1:-1])


def test_fill_from_level_above_chain_and_surface_refusal():
    wet = np.ones((3, 1, 2), dtype=bool)
    out = np.array([[[1.0, 2.0]], [[0.0, 5.0]], [[0.0, 0.0]]])
    need = np.array([[[False, False]], [[True, False]], [[True, True]]])
    cg.fill_from_level_above(out, need, wet)
    np.testing.assert_array_equal(out[:, 0], [[1.0, 2.0], [1.0, 5.0], [1.0, 5.0]])
    with pytest.raises(ValueError, match="surface"):
        cg.fill_from_level_above(np.zeros((2, 1, 1)), np.ones((2, 1, 1), bool),
                                 np.ones((2, 1, 1), bool))


def test_rectilinear_axes_rejects_curvilinear():
    lam = np.tile(np.arange(4.0), (3, 1))
    phi = np.tile(np.arange(3.0)[:, None], (1, 4))
    lam[1, 2] += 1e-9
    with pytest.raises(ValueError, match="rectilinear"):
        cg.rectilinear_axes(lam, phi)


def test_vertical_grid_tolerates_round_off_only(pair):
    f, _, _, _ = pair
    near = {k: f[k] + 1e-10 for k in ("e3t_1d", "gdept_1d", "e3t_0")}
    cg.assert_same_vertical_grid(f, near)     # 1/16 vs 1 deg: 1.05e-10 m
    far = dict(near, e3t_0=f["e3t_0"] + 1e-8)
    with pytest.raises(ValueError, match="e3t_0"):
        cg.assert_same_vertical_grid(f, far)


def test_coarse_grid_tag(pair):
    f, c, _, _ = pair
    assert cg.coarse_grid(f["glamf"]) == ("cg025", "1/4 deg")   # R4 mesh
    assert cg.coarse_grid(c["glamf"]) == ("cg1deg", "1 deg")    # R1 mesh
    with pytest.raises(ValueError, match="zonal spacing"):
        cg.coarse_grid(f["glamf"] * 2.0)


def _write_mesh(path, m):
    nz, ny, nx = m["tmask"].shape
    with netCDF4.Dataset(path, "w") as d:
        for name, size in (("time_counter", 1), ("nav_lev", nz), ("y", ny), ("x", nx)):
            d.createDimension(name, size)
        for k in ("glamf", "gphif"):
            d.createVariable(k, "f8", ("time_counter", "y", "x"))[:] = m[k][None]
        for k in ("e3t_1d", "gdept_1d"):
            d.createVariable(k, "f8", ("time_counter", "nav_lev"))[:] = m[k][None]
        d.createVariable("tmask", "i1", ("time_counter", "nav_lev", "y", "x"))[:] = m["tmask"][None]
        d.createVariable("e3t_0", "f8", ("time_counter", "nav_lev", "y", "x"))[:] = (
            np.broadcast_to(m["e3t_0"][:, None, None], m["tmask"].shape)[None])
        for k in ("nav_lat", "nav_lon"):
            d.createVariable(k, "f4", ("y", "x"))[:] = m[k]


def test_main_writes_month_file(pair, tmp_path):
    f, c, _, _ = pair
    _write_mesh(tmp_path / "fine_mesh.nc", f)
    _write_mesh(tmp_path / "coarse_mesh.nc", c)
    nz, ny, nx = f["tmask"].shape
    src = tmp_path / "MINT_1d_00610101_00610102_grid_T.nc"
    with netCDF4.Dataset(src, "w") as d:
        for name, size in (("time_counter", None), ("deptht", nz), ("y", ny), ("x", nx)):
            d.createDimension(name, size)
        t = d.createVariable("time_counter", "f8", ("time_counter",))
        t.units, t.calendar = "seconds since 1900-01-01 00:00:00", "360_day"
        t[:] = [86400.0, 172800.0]
        d.createVariable("deptht", "f4", ("deptht",))[:] = f["gdept_1d"]
        for k in ("nav_lat", "nav_lon"):
            d.createVariable(k, "f4", ("y", "x"))[:] = f[k]
        for k, val in (("thetao", 4.0), ("so", 35.0)):
            v = d.createVariable(k, "f4", ("time_counter", "deptht", "y", "x"),
                                 fill_value=np.float32(1e20))
            v[:] = np.where(f["tmask"], val, 0.0)[None].repeat(2, axis=0)
        v = d.createVariable("zos", "f4", ("time_counter", "y", "x"),
                             fill_value=np.float32(1e20))
        v[:] = np.where(f["tmask"][0], 0.1, 0.0)[None].repeat(2, axis=0)

    rc = cg.main(["--grid-t", str(src), "--fine-mesh", str(tmp_path / "fine_mesh.nc"),
                  "--coarse-mesh", str(tmp_path / "coarse_mesh.nc"),
                  "--out-dir", str(tmp_path / "out"), "--days", "1"])
    assert rc == 0
    with netCDF4.Dataset(tmp_path / "out" / "MINT_1d_00610101_00610102_state_cg1deg.nc") as o:
        assert o["thetao"].shape == (1,) + c["tmask"].shape
        assert o["zos"].shape == (1,) + c["tmask"].shape[1:]
        assert o["thetao"].dtype == np.float32
        th = np.asarray(o["thetao"][0])
        assert np.max(np.abs(th[c["tmask"]] - 4.0)) < 1e-6
        assert np.all(th[~c["tmask"]] == 0.0)
        assert float(o["time_counter"][0]) == 86400.0
        assert o["time_counter"].calendar == "360_day"
        assert o["coverage"].shape == c["tmask"].shape
        assert o.legoesm_git_commit
        assert o.title.endswith("to the 1 deg mesh")
