"""Direct test of ``scripts/data/coarse_grain_dino_uv.py``.

Synthetic pair built analytically by ``nemo_dino_mesh``: DINO 1/4 deg as the
"fine" mesh, DINO 1 deg as the "coarse" one (as in the T/S test): coincident U
columns plus one seam column, V rows that do not nest, ring columns only partly
covered by fine faces.
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

from scripts.data import coarse_grain_dino_uv as cg  # noqa: E402


def _mesh(nml):
    g, h = nemo_dino_mesh(nml), nemo_dino_hgr(nml)
    m = {k: h[k] for k in ("glamf", "gphif", "glamu", "gphiu", "glamv", "gphiv", "e2u", "e1v")}
    m["umask"] = np.moveaxis(g.umask, -1, 0).astype(bool)          # (z, y, x)
    m["vmask"] = np.moveaxis(g.vmask, -1, 0).astype(bool)
    m["e3u_0"] = g.e3u_0[0, 0].copy()
    m["e3v_0"] = g.e3v_0[0, 0].copy()
    return m


@pytest.fixture(scope="module")
def pair():
    f, c = _mesh(NEMO_DINO_R4), _mesh(NEMO_DINO_R1)
    return f, c, cg.u_operator(f, c), cg.v_operator(f, c)


def _fine_sum_through_coarse_u_faces(u_f, wet_f, len_f, f, c):
    """Fine transport through each coarse U face, summed independently.

    Cumulative transport along latitude, interpolated linearly at the coarse
    row edges (= the latitude-fraction split), at the coincident column.
    """
    fe = cg.cell_edges_rad(f["gphif"][:, 0])
    ce = cg.cell_edges_rad(c["gphif"][:, 0])
    q = np.where(wet_f, u_f * len_f, 0.0)
    cum = np.concatenate([np.zeros(q.shape[:1] + (1,) + q.shape[2:]), np.cumsum(q, axis=1)], axis=1)
    at = np.stack([[np.interp(ce, fe, cum[k, :, i]) for i in range(q.shape[2])]
                   for k in range(q.shape[0])])                    # (z, x_f, ny_c+1)
    cols = [int(np.flatnonzero(np.isclose(f["glamu"][0], lon))[0]) for lon in c["glamu"][0, :-1]]
    return np.diff(at, axis=2)[:, cols, :].transpose(0, 2, 1)      # (z, ny_c, nx_c-1)


def test_u_transport_conserved_where_faces_nest(pair):
    f, c, op, _ = pair
    rng = np.random.default_rng(0)
    u_f = rng.normal(0.0, 0.3, f["umask"].shape)
    wet_len, cov_len = cg.static_lengths(op, f["umask"], c["umask"])
    vel, _ = cg.coarse_velocity(u_f, f["umask"], c["umask"], op, wet_len, cov_len)
    ref = _fine_sum_through_coarse_u_faces(u_f, f["umask"], f["e2u"], f, c)
    full = (c["umask"] & np.isclose(wet_len, cov_len, rtol=1e-12, atol=0.0))[:, :, :-1]
    assert full.sum() > 1000
    got = (vel * c["e2u"])[:, :, :-1]
    scale = np.max(np.abs(ref[full]))
    assert np.max(np.abs(got - ref)[full]) < 1e-12 * scale
    # non-vacuous: without the e2u weighting of the fine faces it fails
    flat = cg.FaceOperator(op.mrow.toarray(), op.mcol_t.T.toarray(),
                           np.broadcast_to(f["e2u"].mean(axis=0), f["e2u"].shape), op.len_c)
    vel0, _ = cg.coarse_velocity(u_f, f["umask"], c["umask"], flat,
                                 *cg.static_lengths(flat, f["umask"], c["umask"]))
    assert np.max(np.abs((vel0 * c["e2u"])[:, :, :-1] - ref)[full]) > 1e-6 * scale


def _div(tu, tv):
    """Horizontal divergence of face transports (z, y, x); i periodic, none south of row 0."""
    tv_south = np.concatenate([np.zeros_like(tv[:, :1]), tv[:, :-1]], axis=1)
    return tu - np.roll(tu, 1, axis=2) + tv - tv_south


def test_coarse_divergence_is_fraction_weighted_fine_divergence(pair):
    f, c, op_u, op_v = pair
    nz = 2
    wet = {k: np.ones((nz,) + f[k].shape[1:], bool) for k in ("umask", "vmask")}
    wet_c = {k: np.ones((nz,) + c[k].shape[1:], bool) for k in ("umask", "vmask")}
    for w in (wet, wet_c):
        w["vmask"][:, -1] = False                  # closed northern boundary
    rng = np.random.default_rng(1)
    u_f, v_f = rng.normal(size=wet["umask"].shape), rng.normal(size=wet["vmask"].shape)
    tu = cg.coarse_velocity(u_f, wet["umask"], wet_c["umask"], op_u,
                            *cg.static_lengths(op_u, wet["umask"], wet_c["umask"]))[0] * c["e2u"]
    tv = cg.coarse_velocity(v_f, wet["vmask"], wet_c["vmask"], op_v,
                            *cg.static_lengths(op_v, wet["vmask"], wet_c["vmask"]))[0] * c["e1v"]
    div_c = _div(tu, tv)
    div_f = _div(np.where(wet["umask"], u_f * f["e2u"], 0.0),
                 np.where(wet["vmask"], v_f * f["e1v"], 0.0))
    # fine T cells to coarse T cells by latitude and longitude overlap fractions
    rows = cg.overlap_fractions(cg.cell_edges_rad(f["gphif"][:, 0]),
                                cg.cell_edges_rad(c["gphif"][:, 0]))
    cols = cg.overlap_fractions(cg.cell_edges_rad(f["glamf"][0]), cg.cell_edges_rad(c["glamf"][0]))
    agg = np.einsum("cj,zji,di->zcd", rows, div_f, cols)
    inner = slice(1, -1)                           # rows 0 and n-1 only partly covered
    scale = np.max(np.abs(tu))
    assert np.max(np.abs(div_c - agg)[:, inner]) < 1e-11 * scale
    # ring columns: their V faces are 1/4 covered (R4 -> R1); the transport is kept
    assert np.max(np.abs(div_c - agg)[:, inner][:, :, [0, -1]]) < 1e-11 * scale


def test_column_map_seam_and_refusal():
    fine = np.arange(0.0, 3.25, 0.25)              # 0 .. 3.0, last is the fine seam
    m = cg.column_map(fine, np.array([0.0, 1.0, 2.0, 4.0]))
    assert [int(np.argmax(r)) for r in m] == [0, 4, 8, fine.size - 1]
    with pytest.raises(ValueError, match="no fine U column"):
        cg.column_map(fine, np.array([0.0, 1.1, 4.0]))


def test_interp_rows_linear_and_out_of_range():
    m = cg.interp_rows(np.array([0.0, 1.0, 3.0]), np.array([0.25, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(m, [[0.75, 0.25, 0], [0, 0.5, 0.5], [0, 0, 1], [0, 0, 0]])


def _write_mesh(path, m):
    nz, ny, nx = m["umask"].shape
    with netCDF4.Dataset(path, "w") as d:
        for name, size in (("time_counter", 1), ("nav_lev", nz), ("y", ny), ("x", nx)):
            d.createDimension(name, size)
        for k in ("glamf", "gphif", "glamu", "gphiu", "glamv", "gphiv", "e2u", "e1v"):
            d.createVariable(k, "f8", ("time_counter", "y", "x"))[:] = m[k][None]
        for k in ("umask", "vmask"):
            d.createVariable(k, "i1", ("time_counter", "nav_lev", "y", "x"))[:] = m[k][None]
        for k in ("e3u_0", "e3v_0"):
            d.createVariable(k, "f8", ("time_counter", "nav_lev", "y", "x"))[:] = (
                np.broadcast_to(m[k][:, None, None], m["umask"].shape)[None])


def _write_src(path, m, var, depth, mask, val):
    nz, ny, nx = m["umask"].shape
    with netCDF4.Dataset(path, "w") as d:
        for name, size in (("time_counter", None), (depth, nz), ("y", ny), ("x", nx)):
            d.createDimension(name, size)
        t = d.createVariable("time_counter", "f8", ("time_counter",))
        t.units, t.calendar = "seconds since 1900-01-01 00:00:00", "360_day"
        t[:] = [86400.0, 172800.0]
        z = d.createVariable(depth, "f4", (depth,))
        z[:] = np.arange(nz)
        z.units = "m"
        v = d.createVariable(var, "f4", ("time_counter", depth, "y", "x"),
                             fill_value=np.float32(1e20))
        v[:] = np.where(m[mask], val, 0.0)[None].repeat(2, axis=0)


def test_main_writes_month_file(pair, tmp_path):
    f, c, _, _ = pair
    _write_mesh(tmp_path / "fine_mesh.nc", f)
    _write_mesh(tmp_path / "coarse_mesh.nc", c)
    su = tmp_path / "MINT_1d_00610101_00610102_grid_U.nc"
    sv = tmp_path / "MINT_1d_00610101_00610102_grid_V.nc"
    _write_src(su, f, "uo", "depthu", "umask", 0.1)
    _write_src(sv, f, "vo", "depthv", "vmask", 0.1)
    rc = cg.main(["--grid-u", str(su), "--grid-v", str(sv),
                  "--fine-mesh", str(tmp_path / "fine_mesh.nc"),
                  "--coarse-mesh", str(tmp_path / "coarse_mesh.nc"),
                  "--out-dir", str(tmp_path / "out"), "--days", "1"])
    assert rc == 0
    assert not list((tmp_path / "out").glob("*.tmp"))
    with netCDF4.Dataset(tmp_path / "out" / "MINT_1d_00610101_00610102_uv_cg1deg.nc") as o:
        u, v = np.asarray(o["uo"][0]), np.asarray(o["vo"][0])
        assert o["uo"].dtype == np.float32 and u.shape == c["umask"].shape
        assert np.all(u[~c["umask"]] == 0.0) and np.all(v[~c["vmask"]] == 0.0)
        cov_v = np.asarray(o["coverage_v"])
        full = c["vmask"] & np.isclose(cov_v, 1.0, atol=1e-6)
        assert np.max(np.abs(v[full] - 0.1)) < 1e-6
        ring = c["vmask"][:, :, 0]                 # 1/4 covered: transport kept
        assert np.max(np.abs(v[:, :, 0][ring] - 0.1 * cov_v[:, :, 0][ring])) < 1e-6
        assert o["uo"].standard_name == "sea_water_x_velocity"
        assert o["vo"].units == "m/s"
        assert o.legoesm_git_commit
        assert float(o["time_counter"][0]) == 86400.0
