"""Coarse-grain daily 1/16 deg DINO (NEMO) T, S and SSH onto the 1/4 deg mesh.

Track B of ``markdown-notes/coarse_graining_full_domain_plan.md`` (group B1):
the coarse states that start and score the legoESM online training windows.

Method (decided by the user; plan D5, D6, D14, D17, D18):

* unfiltered, mask-aware, area-conservative average on each level.  Both DINO
  meshes are exactly rectilinear in lat/lon and share one vertical ladder
  (asserted below), so per-level horizontal-area weights are the volume
  weights.  Overlaps come from the shared kernel
  :func:`legoesm.grids.conservative_regrid.compute_overlap_weights`
  (sin(lat) x lon, ``normalization='dstarea'``); no new overlap code.
* wet coarse cell: ``T_c = R(T_f * m_f) / R(m_f)`` with ``R`` the regrid and
  ``m_f`` the fine wet mask of that level, i.e. the mean over the fine wet area
  under the cell.  ``R(m_f)`` is the coverage field written alongside.  The
  coarse ring columns (only a quarter covered by fine cells) thus take the
  mean of the covered quarter.
* wet coarse cell with no wet fine cell beneath (deep, on the bowl slope):
  copy the value from the level above in the same column.
* dry coarse cell: exactly 0.
* output float32, NEMO order ``(t, z, y, x)``; ``zos`` ``(t, y, x)``.

One monthly ``grid_T`` file in, one monthly file out.  ``--days N`` limits the
run to the first N days (the one-day check).  Every day prints the content
check: the coarse wet content plus the content landing in dry coarse cells must
equal the fine wet content, with fine areas taken from the fine edges.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import netCDF4
import numpy as np
from legoesm.grids.conservative_regrid import (
    apply_conservative_regrid,
    compute_overlap_weights,
)
from legoesm.io.git_provenance import git_provenance

# NEMO variables coarse-grained here: name -> number of spatial dims.
_STATE_VARS = {"thetao": 3, "so": 3, "zos": 2}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def rectilinear_axes(glamf: np.ndarray, gphif: np.ndarray):
    """1-D F-point longitudes ``(x,)`` and latitudes ``(y,)`` of a rectilinear mesh.

    Raises if the 2-D arrays are not exactly constant along the other axis:
    separable overlap weights are only valid on such a mesh.
    """
    glamf = np.asarray(glamf, dtype=np.float64)
    gphif = np.asarray(gphif, dtype=np.float64)
    if np.any(glamf != glamf[0:1, :]) or np.any(gphif != gphif[:, 0:1]):
        raise ValueError("mesh is not exactly rectilinear in lat/lon; "
                         "separable overlap weights do not apply")
    return glamf[0, :], gphif[:, 0]


def cell_edges_rad(f_axis_deg: np.ndarray) -> np.ndarray:
    """Cell edges in radians from 1-D F-point coordinates (degrees).

    Cell ``i`` spans ``f[i-1]..f[i]`` (F is the north/east corner in NEMO); the
    missing first edge is extrapolated by one spacing, as plan item G2 does.
    """
    f = np.asarray(f_axis_deg, dtype=np.float64)
    return np.deg2rad(np.concatenate([[f[0] - (f[1] - f[0])], f]))


def overlap_weights(fine_glamf, fine_gphif, coarse_glamf, coarse_gphif):
    """Fine -> coarse ``dstarea`` overlap weights from 2-D F-point arrays."""
    flon, flat = rectilinear_axes(fine_glamf, fine_gphif)
    clon, clat = rectilinear_axes(coarse_glamf, coarse_gphif)
    return compute_overlap_weights(
        cell_edges_rad(flat), cell_edges_rad(flon),
        cell_edges_rad(clat), cell_edges_rad(clon),
        normalization="dstarea",
    )


def cell_areas(glamf, gphif) -> np.ndarray:
    """Cell areas ``(y, x)`` on the unit sphere, in the overlap kernel's measure."""
    lon, lat = rectilinear_axes(glamf, gphif)
    s = np.sin(cell_edges_rad(lat))
    return np.outer(np.diff(s), np.diff(cell_edges_rad(lon)))


def regrid(field: np.ndarray, weights) -> np.ndarray:
    """``R(field)`` in float64; leading axes (levels) are vmapped."""
    return np.asarray(apply_conservative_regrid(
        jax.numpy.asarray(field, dtype=jax.numpy.float64), weights))


def assert_same_vertical_grid(fine: dict, coarse: dict) -> None:
    """Both meshes must share one horizontally uniform vertical ladder.

    Per-level area weights equal volume weights only then (plan, B1).
    """
    for name in ("e3t_1d", "gdept_1d", "e3t_0"):
        if not np.array_equal(fine[name], coarse[name]):
            raise ValueError(f"{name} differs between the fine and coarse meshes")


# --------------------------------------------------------------------------
# the coarse-graining
# --------------------------------------------------------------------------

def fill_from_level_above(out: np.ndarray, need: np.ndarray, wet_c: np.ndarray):
    """Copy the level above into wet coarse cells with no wet fine cell (in place).

    ``out``, ``need``, ``wet_c`` are ``(z, y, x)``; levels run top to bottom, so
    a chain of such cells inherits from the first level that had fine water.
    """
    if need[0].any():
        raise ValueError(
            f"{int(need[0].sum())} wet coarse surface cell(s) have no wet fine "
            "cell beneath and no level above to copy from")
    for k in range(1, out.shape[0]):
        if not need[k].any():
            continue
        if np.any(need[k] & ~wet_c[k - 1]):
            raise ValueError(f"level {k}: a cell to fill sits under a dry cell")
        out[k][need[k]] = out[k - 1][need[k]]
    return out


def coarse_grain(field_f, tmask_f, tmask_c, coverage, weights) -> np.ndarray:
    """Mask-aware coarse field ``(z, y, x)`` (float64) from fine ``(z, y, x)``.

    ``coverage = R(tmask_f)`` (precomputed once per run).  Fine land values are
    multiplied by 0, so whatever the file holds there does not matter.
    """
    wet_f = tmask_f.astype(bool)
    wet_c = tmask_c.astype(bool)
    num = regrid(np.where(wet_f, field_f, 0.0), weights)
    has_fine = wet_c & (coverage > 0.0)
    out = np.zeros_like(num)
    out[has_fine] = num[has_fine] / coverage[has_fine]
    return fill_from_level_above(out, wet_c & ~has_fine, wet_c)


def content_check(field_f, out_c, tmask_f, tmask_c, coverage, weights,
                  area_f, area_c, e3) -> dict:
    """Content closure and wet means, float64, summed over levels with ``e3``.

    ``fine`` = fine wet content; ``lost`` = content landing in dry coarse cells;
    ``coarse`` = coarse wet content counted with the fine wet area under each
    cell (``area_c * coverage``).  Construction makes ``coarse + lost = fine``
    up to round-off and float32 output; ``rel_closure`` measures that.
    ``mean_*`` are the volume means after removing the lost part; ``plain_*``
    weights the coarse cells by their full wet volume instead.
    """
    wet_f = tmask_f.astype(bool)
    wet_c = tmask_c.astype(bool)
    e3 = np.asarray(e3, dtype=np.float64)[:, None, None]
    fv = np.where(wet_f, np.asarray(field_f, dtype=np.float64), 0.0)
    num = regrid(fv, weights)
    fine = float(np.sum(e3 * area_f * fv))
    vol_f = float(np.sum(e3 * area_f * wet_f))
    lost = float(np.sum(np.where(wet_c, 0.0, e3 * area_c * num)))
    vol_lost = float(np.sum(np.where(wet_c, 0.0, e3 * area_c * coverage)))
    oc = np.asarray(out_c, dtype=np.float64)
    w_eff = np.where(wet_c, e3 * area_c * coverage, 0.0)
    coarse = float(np.sum(w_eff * oc))
    w_plain = np.where(wet_c, e3 * area_c, 0.0)
    mean_c = coarse / float(np.sum(w_eff))
    mean_f = (fine - lost) / (vol_f - vol_lost)
    return dict(
        fine=fine, lost=lost, coarse=coarse,
        rel_closure=abs(coarse + lost - fine) / abs(fine),
        lost_volume_fraction=vol_lost / vol_f,
        mean_coarse=mean_c, mean_fine_corrected=mean_f,
        rel_mean_diff=abs(mean_c - mean_f) / abs(mean_f),
        plain_mean_coarse=float(np.sum(w_plain * oc) / np.sum(w_plain)),
        plain_mean_fine=fine / vol_f,
        n_nonfinite=int(np.count_nonzero(~np.isfinite(oc))),
        max_abs_dry=float(np.max(np.abs(np.where(wet_c, 0.0, oc)))),
    )


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def read_mesh(path: Path) -> dict:
    """The mesh fields this script needs, NEMO order, float64 / bool."""
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        m = {k: np.asarray(d[k][0], dtype=np.float64)
             for k in ("glamf", "gphif", "e3t_1d", "gdept_1d")}
        m["tmask"] = np.asarray(d["tmask"][0]).astype(bool)
        e3t0 = np.empty(m["e3t_1d"].shape)
        for k in range(e3t0.size):           # per level: 3-D read is 735 MB on 1/16
            lev = np.asarray(d["e3t_0"][0, k], dtype=np.float64)
            if np.any(lev != lev.flat[0]):
                raise ValueError(f"{path}: e3t_0 level {k} is not horizontally uniform")
            e3t0[k] = lev.flat[0]
        m["e3t_0"] = e3t0
        m["nav_lat"] = np.asarray(d["nav_lat"][:])
        m["nav_lon"] = np.asarray(d["nav_lon"][:])
    return m


def read_day(ds: netCDF4.Dataset, name: str, t: int, wet: np.ndarray) -> np.ndarray:
    """One day of ``name`` (partial read); refuse fill or non-finite on wet cells."""
    v = ds[name]
    a = np.asarray(v[t], dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    fill = getattr(v, "_FillValue", None)
    bad = ~np.isfinite(a) if fill is None else (~np.isfinite(a) | (a == fill))
    if np.any(bad & wet):
        raise ValueError(f"{name} day {t}: {int((bad & wet).sum())} wet cell(s) "
                         "hold fill or non-finite values")
    return a


def create_output(path: Path, src: netCDF4.Dataset, coarse: dict, coverage,
                  lost_frac, attrs: dict) -> netCDF4.Dataset:
    nz, ny, nx = coverage.shape
    out = netCDF4.Dataset(path, "w", format="NETCDF4")
    out.createDimension("time_counter", None)
    out.createDimension("deptht", nz)
    out.createDimension("y", ny)
    out.createDimension("x", nx)

    def copy_attrs(dst, srcv, keys):
        for a in keys:
            if a in srcv.ncattrs():
                dst.setncattr(a, srcv.getncattr(a))

    for name in ("nav_lat", "nav_lon"):
        v = out.createVariable(name, "f4", ("y", "x"))
        v[:] = coarse[name]
        copy_attrs(v, src[name], ("long_name", "units"))
    v = out.createVariable("deptht", "f4", ("deptht",))
    v[:] = src["deptht"][:]
    copy_attrs(v, src["deptht"], ("long_name", "units"))
    v = out.createVariable("time_counter", "f8", ("time_counter",))
    copy_attrs(v, src["time_counter"], ("long_name", "units", "calendar"))
    for name, nd in _STATE_VARS.items():
        dims = ("time_counter", "deptht", "y", "x") if nd == 3 else ("time_counter", "y", "x")
        v = out.createVariable(name, "f4", dims)
        copy_attrs(v, src[name], ("long_name", "units"))
    v = out.createVariable("coverage", "f4", ("deptht", "y", "x"))
    v.long_name = "fraction of the coarse cell area covered by wet fine cells"
    v[:] = coverage
    v = out.createVariable("lost_volume_fraction", "f4", ("deptht",))
    v.long_name = ("fine wet volume under dry coarse cells / fine wet volume, "
                   "per level")
    v[:] = lost_frac
    for k, val in attrs.items():
        out.setncattr(k, val)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--grid-t", type=Path, required=True,
                   help="one monthly 1/16 deg MINT_1d_*_grid_T.nc file")
    p.add_argument("--fine-mesh", type=Path, required=True,
                   help="1/16 deg mesh_mask (EXP16/mesh_mask_exp16.nc)")
    p.add_argument("--coarse-mesh", type=Path, required=True,
                   help="1/4 deg mesh_mask (EXP4/mesh_mask_025.nc)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--days", type=int, default=None,
                   help="process only the first N days (default: whole file)")
    return p


def output_name(grid_t: Path) -> str:
    if "grid_T" not in grid_t.name:
        raise ValueError(f"{grid_t.name}: expected a *grid_T*.nc file")
    return grid_t.name.replace("grid_T", "state_cg025")


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Weights and sums in float64; the output is float32 (plan D14).
    jax.config.update("jax_enable_x64", True)

    t0 = time.perf_counter()
    fine = read_mesh(args.fine_mesh)
    coarse = read_mesh(args.coarse_mesh)
    assert_same_vertical_grid(fine, coarse)
    weights = overlap_weights(fine["glamf"], fine["gphif"],
                              coarse["glamf"], coarse["gphif"])
    area_f = cell_areas(fine["glamf"], fine["gphif"])
    area_c = cell_areas(coarse["glamf"], coarse["gphif"])
    tm_f, tm_c = fine["tmask"], coarse["tmask"]
    coverage = regrid(tm_f, weights)
    lost_area = np.sum(np.where(tm_c, 0.0, area_c * coverage), axis=(1, 2))
    wet_area_f = np.sum(area_f * tm_f, axis=(1, 2))
    with np.errstate(invalid="ignore"):      # NaN on a level with no water at all
        lost_frac = lost_area / wet_area_f
    n_fill = int(np.count_nonzero(tm_c & (coverage == 0.0)))
    print(f"setup {time.perf_counter() - t0:.1f} s: {weights.weights.size} overlaps, "
          f"{n_fill} wet coarse cells with no wet fine cell (filled from above), "
          f"min nonzero coverage {coverage[coverage > 0].min():.3e}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / output_name(args.grid_t)
    prov = git_provenance(__file__)
    attrs = dict(
        title="DINO 1/16 deg T, S, SSH coarse-grained to the 1/4 deg mesh",
        method=("unfiltered mask-aware area-conservative mean per level "
                "(R(T*m)/R(m), compute_overlap_weights dstarea); wet coarse "
                "cells with no wet fine cell copy the level above; dry cells 0"),
        source_file=str(args.grid_t.resolve()),
        fine_mesh=str(args.fine_mesh.resolve()),
        coarse_mesh=str(args.coarse_mesh.resolve()),
        command=" ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        legoesm_git_commit=prov.commit, legoesm_git_dirty=int(prov.dirty),
        n_filled_from_above=n_fill,
        lost_volume_fraction_total=float(np.sum(coarse["e3t_0"] * lost_area)
                                         / np.sum(coarse["e3t_0"] * wet_area_f)),
    )

    with netCDF4.Dataset(args.grid_t) as src:
        src.set_auto_mask(False)
        n_days = src.dimensions["time_counter"].size
        if args.days is not None:
            if not 1 <= args.days <= n_days:
                raise ValueError(f"--days {args.days} outside 1..{n_days}")
            n_days = args.days
        out = create_output(out_path, src, coarse, coverage, lost_frac, attrs)
        try:
            for t in range(n_days):
                td = time.perf_counter()
                out["time_counter"][t] = src["time_counter"][t]
                for name, nd in _STATE_VARS.items():
                    lev = slice(None) if nd == 3 else slice(0, 1)
                    e3 = coarse["e3t_0"][lev] if nd == 3 else np.ones(1)
                    f = read_day(src, name, t, tm_f[lev])
                    c = coarse_grain(f, tm_f[lev], tm_c[lev], coverage[lev], weights)
                    chk = content_check(f, c.astype(np.float32), tm_f[lev], tm_c[lev],
                                        coverage[lev], weights, area_f, area_c, e3)
                    if chk["n_nonfinite"] or chk["max_abs_dry"] != 0.0:
                        raise RuntimeError(f"{name} day {t}: {chk}")
                    out[name][t] = c.astype(np.float32) if nd == 3 else c[0].astype(np.float32)
                    print(f"day {t} {name}: rel_closure {chk['rel_closure']:.3e} "
                          f"mean_coarse {chk['mean_coarse']:.9g} "
                          f"mean_fine_corrected {chk['mean_fine_corrected']:.9g} "
                          f"rel_mean_diff {chk['rel_mean_diff']:.3e} "
                          f"plain_mean_coarse {chk['plain_mean_coarse']:.9g} "
                          f"plain_mean_fine {chk['plain_mean_fine']:.9g} "
                          f"lost {chk['lost']:.6g} of {chk['fine']:.6g}", flush=True)
                print(f"day {t}: {time.perf_counter() - td:.1f} s", flush=True)
        finally:
            out.close()
    print(f"wrote {out_path} ({n_days} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
