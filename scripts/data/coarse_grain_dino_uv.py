"""Coarse-grain daily 1/16 deg DINO (NEMO) u, v onto the 1 deg (or 1/4 deg) mesh.

Extends Track B (``coarse_grain_dino_state.py``: T, S, SSH) with the 3-D
velocities, so that a legoESM run can start from a coarse state in balance.

Method: transport-conserving C-grid face averaging, unfiltered (the velocity
analogue of Track B's conservative average).  For each level and each coarse
face (U: east face of T(j, i), length ``e2u``; V: north face, length ``e1v``):

* fine face transports ``q_f = vel_f * L_f * mask_f`` (``L_f`` = e2u or e1v;
  ``e3`` is horizontally uniform per level and equal on both meshes, asserted,
  so it cancels level by level);
* ``Q = Mrow @ q_f @ Mcol.T`` collects them onto the coarse face:
  - U: rows by the latitude overlap of fine T rows with the coarse T row
    (fraction of each fine face inside the coarse face, linear in latitude);
    columns by coincidence: coarse U column i sits on the fine U column at the
    same longitude; the last coarse column is the periodic seam (east face of
    the east ring column) and takes the fine seam column (the last one).
  - V: coarse V rows do not lie on fine V rows (on DINO they sit half-way, on
    fine T latitudes), so the fine transport is interpolated linearly in
    latitude between the two bracketing fine V rows.  This is the overlap rule:
    a fine T row cut by the coarse face gives the same fractions to the
    U faces, so coarse cell divergences are fraction-weighted sums of fine
    ones.  Columns by the longitude overlap of fine with coarse V faces.
* ``vel_c = Q / (L_c * wet)``, with ``wet = W / C`` the wet fraction of the
  covered fine length (``W`` = ``Mrow @ (L_f * mask_f) @ Mcol.T``, ``C`` the same
  without the mask).  A fully wet coarse face thus carries exactly the fine
  transport (``vel_c * L_c = Q``); at coasts the velocity is renormalised over
  the wet fine part (as Track B renormalises T); where the fine faces cover
  only part of the coarse face (ring-column V faces: 1/16 of the width at
  1 deg) the transport, not the velocity, is kept.
* wet coarse face with no wet fine face beneath: copy the level above (plan D6).
* dry coarse face: exactly 0.  Output float32, NEMO order ``(t, z, y, x)``.

One monthly ``grid_U`` + ``grid_V`` pair in, one monthly file out.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import netCDF4
import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.grids import conservative_regrid  # noqa: E402
from legoesm.grids.conservative_regrid import compute_overlap_weights  # noqa: E402
from legoesm.io.git_provenance import git_provenance  # noqa: E402

from scripts.data.coarse_grain_dino_state import (  # noqa: E402
    _VERTICAL_ATOL_M,
    cell_edges_rad,
    coarse_grid,
    fill_from_level_above,
    read_day,
    rectilinear_axes,
)

# One latitude band: the kernel's longitude axis is then a plain 1-D overlap.
_ONE_BAND = np.array([0.0, 0.1])


# --------------------------------------------------------------------------
# face operators
# --------------------------------------------------------------------------

def overlap_fractions(src_edges, dst_edges) -> np.ndarray:
    """``(n_dst, n_src)``: fraction of each source interval inside each destination one.

    1-D, linear in the coordinate; built with the shared overlap kernel.
    """
    src_edges = np.asarray(src_edges, dtype=np.float64)
    dst_edges = np.asarray(dst_edges, dtype=np.float64)
    w = compute_overlap_weights(_ONE_BAND, src_edges, _ONE_BAND, dst_edges,
                                normalization="dstarea")   # overlap / dst length
    if w.weights.dtype != np.float64:
        raise RuntimeError("overlap weights are not float64: enable jax_enable_x64")
    m = np.zeros((dst_edges.size - 1, src_edges.size - 1))
    np.add.at(m, (np.asarray(w.dst_idx_flat), np.asarray(w.src_idx_flat)),
              np.asarray(w.weights))
    return m * np.diff(dst_edges)[:, None] / np.diff(src_edges)[None, :]


def column_map(fine_lon, coarse_lon) -> np.ndarray:
    """``(n_c, n_f)`` selection of the fine U column under each coarse U column.

    Coincident longitude (to 1e-9 deg), except the last coarse column: the
    periodic seam, which takes the fine seam (last) column.
    """
    m = np.zeros((coarse_lon.size, fine_lon.size))
    for i, lon in enumerate(coarse_lon):
        hit = np.flatnonzero(np.abs(fine_lon - lon) < 1e-9)
        if hit.size == 1:
            m[i, hit[0]] = 1.0
        elif i == coarse_lon.size - 1:
            m[i, -1] = 1.0
        else:
            raise ValueError(f"coarse U column {i} (lon {lon}) has no fine U column")
    return m


def interp_rows(fine_lat, coarse_lat) -> np.ndarray:
    """``(n_c, n_f)`` linear interpolation in latitude from fine to coarse V rows.

    A coarse row outside the fine rows gets a zero row (it must be dry; the
    caller checks).
    """
    m = np.zeros((coarse_lat.size, fine_lat.size))
    k = np.searchsorted(fine_lat, coarse_lat, side="right") - 1
    for j, (kk, lat) in enumerate(zip(k, coarse_lat)):
        if 0 <= kk < fine_lat.size - 1:
            a = (lat - fine_lat[kk]) / (fine_lat[kk + 1] - fine_lat[kk])
            m[j, kk], m[j, kk + 1] = 1.0 - a, a
        elif kk == fine_lat.size - 1 and lat == fine_lat[-1]:
            m[j, kk] = 1.0
    return m


class FaceOperator:
    """``Q = Mrow @ q @ Mcol.T`` per level, with sparse factors."""

    def __init__(self, mrow, mcol, len_f, len_c):
        self.mrow = sp.csr_matrix(mrow)
        self.mcol_t = sp.csr_matrix(np.asarray(mcol).T)
        self.len_f = np.asarray(len_f, dtype=np.float64)   # (ny_f, nx_f)
        self.len_c = np.asarray(len_c, dtype=np.float64)   # (ny_c, nx_c)

    def apply(self, q):
        q = np.asarray(q, dtype=np.float64)
        nz, ny, nx = q.shape
        cols = (self.mcol_t.T @ q.reshape(nz * ny, nx).T).T       # (nz*ny, nx_c)
        cols = cols.reshape(nz, ny, -1).transpose(1, 0, 2).reshape(ny, -1)
        out = (self.mrow @ cols).reshape(self.mrow.shape[0], nz, -1)
        return out.transpose(1, 0, 2)                              # (nz, ny_c, nx_c)


def u_operator(fine: dict, coarse: dict) -> FaceOperator:
    flon, flat = rectilinear_axes(fine["glamf"], fine["gphif"])
    clon, clat = rectilinear_axes(coarse["glamf"], coarse["gphif"])
    return FaceOperator(overlap_fractions(cell_edges_rad(flat), cell_edges_rad(clat)),
                        column_map(flon, clon), fine["e2u"], coarse["e2u"])


def v_operator(fine: dict, coarse: dict) -> FaceOperator:
    flon, flat = rectilinear_axes(fine["glamf"], fine["gphif"])
    clon, clat = rectilinear_axes(coarse["glamf"], coarse["gphif"])
    return FaceOperator(interp_rows(np.deg2rad(flat), np.deg2rad(clat)),
                        overlap_fractions(cell_edges_rad(flon), cell_edges_rad(clon)),
                        fine["e1v"], coarse["e1v"])


def static_lengths(op: FaceOperator, wet_f, wet_c):
    """Wet and covered fine length on each coarse face ``(z, y, x)``, ``(1, y, x)``."""
    wet_len = op.apply(np.where(wet_f, op.len_f, 0.0))
    cov_len = op.apply(op.len_f[None])
    if np.any(wet_c & (cov_len <= 0.0)):
        raise ValueError(f"{int((wet_c & (cov_len <= 0)).sum())} wet coarse face(s) "
                         "have no fine face at all beneath")
    return wet_len, cov_len


def coarse_velocity(vel_f, wet_f, wet_c, op: FaceOperator, wet_len, cov_len):
    """Coarse velocity ``(z, y, x)`` float64 and the coarse face transport per unit e3.

    Returns ``(vel_c, q_c)``; ``q_c`` is the collected fine transport ``Q``
    (before any renormalisation or fill), for the diagnostics.
    """
    q_c = op.apply(np.where(wet_f, vel_f * op.len_f, 0.0))
    has = wet_c & (wet_len > 0.0)
    out = np.zeros_like(q_c)
    # vel = Q / (L_c * W / C)
    out[has] = (q_c * cov_len / (op.len_c * np.where(has, wet_len, 1.0)))[has]
    return fill_from_level_above(out, wet_c & ~has, wet_c), q_c


def transport_budget(vel_c, q_c, wet_c, wet_len, cov_len, op, e3) -> dict:
    """Where the coarse transport differs from the collected fine one (float64).

    ``total`` = sum |Q e3| over all coarse faces; ``lost`` = on dry coarse
    faces; ``renorm`` = |coarse - Q| on wet faces whose fine part is partly
    dry; ``filled`` = coarse transport on faces filled from above.
    """
    e3 = np.asarray(e3, dtype=np.float64)[:, None, None]
    t_c = vel_c * op.len_c * e3
    q = q_c * e3
    has = wet_c & (wet_len > 0.0)
    partial = has & (wet_len < cov_len * (1.0 - 1e-12))
    total = float(np.sum(np.abs(q)))
    return dict(
        total=total,
        lost=float(np.sum(np.abs(np.where(wet_c, 0.0, q)))) / total,
        renorm=float(np.sum(np.abs(np.where(partial, t_c - q, 0.0)))) / total,
        filled=float(np.sum(np.abs(np.where(wet_c & ~has, t_c, 0.0)))) / total,
        full_wet_max_rel=float(np.max(np.abs(np.where(has & ~partial, t_c - q, 0.0)))
                               / max(float(np.max(np.abs(q))), 1e-300)),
        n_nonfinite=int(np.count_nonzero(~np.isfinite(vel_c))),
        max_abs_dry=float(np.max(np.abs(np.where(wet_c, 0.0, vel_c)))),
    )


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def read_mesh(path: Path) -> dict:
    """The mesh fields this script needs, NEMO order, float64 / bool."""
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        m = {k: np.asarray(d[k][0], dtype=np.float64)
             for k in ("glamf", "gphif", "glamu", "gphiu", "glamv", "gphiv", "e2u", "e1v")}
        for k in ("umask", "vmask"):
            m[k] = np.asarray(d[k][0]).astype(bool)
        for k in ("e3u_0", "e3v_0"):
            col = np.empty(d[k].shape[1])
            for lev in range(col.size):   # per level: a 3-D read is 735 MB on 1/16
                a = np.asarray(d[k][0, lev], dtype=np.float64)
                if np.any(a != a.flat[0]):
                    raise ValueError(f"{path}: {k} level {lev} is not horizontally uniform")
                col[lev] = a.flat[0]
            m[k] = col
    return m


def assert_same_e3(fine: dict, coarse: dict) -> None:
    for k in ("e3u_0", "e3v_0"):
        a, b = fine[k], coarse[k]
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=_VERTICAL_ATOL_M):
            raise ValueError(f"{k} differs between the fine and coarse meshes")


def output_name(grid_u: Path, tag: str) -> str:
    if "grid_U" not in grid_u.name:
        raise ValueError(f"{grid_u.name}: expected a *grid_U*.nc file")
    return grid_u.name.replace("grid_U", f"uv_{tag}")


_VARS = {
    "uo": dict(src="uo", depth="depthu", point="U", lon="glamu", lat="gphiu",
               standard_name="sea_water_x_velocity",
               long_name="ocean current along i-axis, coarse-grained",
               location="U point: east face of T cell (j, i)", cov="coverage_u"),
    "vo": dict(src="vo", depth="depthv", point="V", lon="glamv", lat="gphiv",
               standard_name="sea_water_y_velocity",
               long_name="ocean current along j-axis, coarse-grained",
               location="V point: north face of T cell (j, i)", cov="coverage_v"),
}


def create_output(path, src_u, src_v, coarse, coverage, attrs):
    nz, ny, nx = coverage["uo"].shape
    out = netCDF4.Dataset(path, "w", format="NETCDF4")
    out.createDimension("time_counter", None)
    out.createDimension("depthu", nz)
    out.createDimension("depthv", nz)
    out.createDimension("y", ny)
    out.createDimension("x", nx)
    def copy_attrs(dst, srcv, keys):
        for a in keys:
            if a in srcv.ncattrs():
                dst.setncattr(a, srcv.getncattr(a))

    t = out.createVariable("time_counter", "f8", ("time_counter",))
    copy_attrs(t, src_u["time_counter"],
               ("axis", "standard_name", "long_name", "units", "calendar", "time_origin"))
    for name, spec in _VARS.items():
        src = src_u if name == "uo" else src_v
        d = out.createVariable(spec["depth"], "f4", (spec["depth"],))
        d[:] = src[spec["depth"]][:]
        copy_attrs(d, src[spec["depth"]], ("long_name", "units", "positive"))
        for c, std in ((spec["lon"], "longitude"), (spec["lat"], "latitude")):
            v = out.createVariable(c, "f8", ("y", "x"))
            v[:] = coarse[c]
            v.standard_name = std
            v.units = "degrees_east" if std == "longitude" else "degrees_north"
            v.long_name = f"{std} of the coarse {spec['point']} points"
        v = out.createVariable(name, "f4", ("time_counter", spec["depth"], "y", "x"),
                               fill_value=False, chunksizes=(1, nz, ny, nx))
        v.standard_name = spec["standard_name"]
        v.long_name = spec["long_name"]
        v.units = "m/s"
        v.grid_location = spec["location"]
        v.coordinates = f"time_counter {spec['depth']} {spec['lat']} {spec['lon']}"
        v.comment = ("0 on dry coarse faces (mesh umask/vmask); no _FillValue. "
                     "Transport through a wet face = value * e2u (or e1v) * e3u_0 "
                     "(or e3v_0) of the coarse mesh.")
        v = out.createVariable(spec["cov"], "f4", (spec["depth"], "y", "x"), fill_value=False)
        v[:] = coverage[name]
        v.units = "1"
        v.long_name = (f"wet fine face length under the coarse {spec['point']} face / "
                       "coarse face length")
        v.coordinates = f"{spec['depth']} {spec['lat']} {spec['lon']}"
    for k, val in attrs.items():
        out.setncattr(k, val)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--grid-u", type=Path, required=True,
                   help="one monthly 1/16 deg MINT_1d_*_grid_U.nc file")
    p.add_argument("--grid-v", type=Path, required=True,
                   help="the matching MINT_1d_*_grid_V.nc file")
    p.add_argument("--fine-mesh", type=Path, required=True,
                   help="1/16 deg mesh_mask (EXP16/mesh_mask_exp16.nc)")
    p.add_argument("--coarse-mesh", type=Path, required=True,
                   help="1/4 or 1 deg mesh_mask (EXP4/mesh_mask_025.nc, mesh_mask_1deg.nc)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--days", type=int, default=None,
                   help="process only the first N days (default: whole file)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)    # float64 overlap weights
    if args.grid_u.name.replace("grid_U", "grid_V") != args.grid_v.name:
        raise ValueError(f"{args.grid_u.name} and {args.grid_v.name} are not one month's pair")

    t0 = time.perf_counter()
    fine = read_mesh(args.fine_mesh)
    coarse = read_mesh(args.coarse_mesh)
    assert_same_e3(fine, coarse)
    ops = {"uo": u_operator(fine, coarse), "vo": v_operator(fine, coarse)}
    masks = {"uo": ("umask", "e3u_0"), "vo": ("vmask", "e3v_0")}
    lens, coverage, n_fill = {}, {}, {}
    for name, op in ops.items():
        wet_f, wet_c = fine[masks[name][0]], coarse[masks[name][0]]
        lens[name] = static_lengths(op, wet_f, wet_c)
        coverage[name] = lens[name][0] / op.len_c
        n_fill[name] = int(np.count_nonzero(wet_c & (lens[name][0] <= 0.0)))
    print(f"setup {time.perf_counter() - t0:.1f} s: wet coarse faces with no wet fine "
          f"face (filled from above): u {n_fill['uo']}, v {n_fill['vo']}", flush=True)

    tag, label = coarse_grid(coarse["glamf"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / output_name(args.grid_u, tag)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    prov = git_provenance(__file__)
    kprov = git_provenance(conservative_regrid.__file__)
    attrs = dict(
        title=f"DINO 1/16 deg u, v coarse-grained to the {label} mesh",
        method=("unfiltered transport-conserving C-grid face average per level: "
                "fine face transports collected by latitude overlap (U rows), "
                "coincident columns (U; last coarse column = periodic seam = last "
                "fine column), linear latitude interpolation between bracketing fine "
                "V rows (V rows) and longitude overlap (V columns); velocity = "
                "transport / (coarse face length * wet fraction of the covered fine "
                "length); wet coarse faces with no wet fine face copy the level "
                "above; dry faces 0"),
        source_file_u=str(args.grid_u.resolve()),
        source_file_v=str(args.grid_v.resolve()),
        fine_mesh=str(args.fine_mesh.resolve()),
        coarse_mesh=str(args.coarse_mesh.resolve()),
        command=" ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        legoesm_git_commit=prov.commit, legoesm_git_dirty=int(prov.dirty),
        legoesm_package_git_commit=kprov.commit,
        n_filled_from_above_u=n_fill["uo"], n_filled_from_above_v=n_fill["vo"],
    )

    with netCDF4.Dataset(args.grid_u) as su, netCDF4.Dataset(args.grid_v) as sv:
        su.set_auto_mask(False)
        sv.set_auto_mask(False)
        n_days = su.dimensions["time_counter"].size
        if not np.array_equal(su["time_counter"][:], sv["time_counter"][:]):
            raise ValueError("grid_U and grid_V time axes differ")
        if args.days is not None:
            if not 1 <= args.days <= n_days:
                raise ValueError(f"--days {args.days} outside 1..{n_days}")
            n_days = args.days
        out = create_output(tmp_path, su, sv, coarse, coverage, attrs)
        try:
            for t in range(n_days):
                td = time.perf_counter()
                out["time_counter"][t] = su["time_counter"][t]
                for name, src in (("uo", su), ("vo", sv)):
                    mask, e3 = masks[name]
                    wet_f, wet_c = fine[mask], coarse[mask]
                    f = read_day(src, name, t, wet_f)
                    c, q = coarse_velocity(f, wet_f, wet_c, ops[name], *lens[name])
                    b = transport_budget(c, q, wet_c, *lens[name], ops[name], coarse[e3])
                    if b["n_nonfinite"] or b["max_abs_dry"] != 0.0:
                        raise RuntimeError(f"{name} day {t}: {b}")
                    out[name][t] = c.astype(np.float32)
                    print(f"day {t} {name}: |transport| {b['total']:.6g} m3/s; fractions: "
                          f"lost {b['lost']:.3e} renorm {b['renorm']:.3e} filled "
                          f"{b['filled']:.3e}; full-wet max rel {b['full_wet_max_rel']:.2e}; "
                          f"min/max/mean {c.min():.4f} {c.max():.4f} "
                          f"{c[wet_c].mean():.3e} m/s", flush=True)
                print(f"day {t}: {time.perf_counter() - td:.1f} s", flush=True)
        except BaseException:
            out.close()
            tmp_path.unlink(missing_ok=True)
            raise
        out.close()
    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({n_days} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
