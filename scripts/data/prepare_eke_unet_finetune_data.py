"""Prepare the 1 deg data for fine-tuning the DINO EKE U-Net (spec sections 4, 7).

Runs in the legoESM venv (JAX, no TensorFlow), on CPU::

    JAX_PLATFORMS=cpu python scripts/data/prepare_eke_unet_finetune_data.py \
        --config scripts/experiment/dino/eke_unet_finetune_1deg.yaml

Reads the Track A 1 deg pairs (read only), runs the checks 1, 2, 3, 4 and 9 of
the spec (a failed check stops the run), and writes under ``out_dir`` of the
config, each under a temporary name and then renamed:

* ``finetune_data_1deg.npz``: ``X`` (days, 244, 104) float32, the network input of
  :func:`legoesm.ml.eke_unet.network_input` with the config's ``vor_norm``;
  ``y`` (days, 199, 52) float32 = ln(fine_ke) on the scored cells, 0 elsewhere;
  ``scored``, ``tmask``, ``area``, ``lat``, ``lon``, the three index arrays,
  ``clim`` (train-mean map of ln(fine_ke), the D-c baseline) and ``meta`` (JSON:
  git hash, config, SHA-256 of the four input files).
  Beyond the spec's list: ``vor_over_f`` (days, 199, 52) float32, the input to
  ``predict_eke`` (the scoring script needs it for any normalisation);
  ``y_ring`` (days, 199, 2), ln(fine_ke) on the wet cells of the dropped columns
  (they are scored separately, for information); ``crop`` = the field's rows and
  columns inside ``X`` (the TensorFlow script has no JAX to ask).
* ``base_1degstats.npz``: D-b, a copy of the base weights with ``vor_norm``
  replaced by the config's.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "scripts" / "experiment" / "dino" / "eke_unet_finetune_1deg.yaml"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check(name, ok, detail=""):
    """Print the result of a spec section 7 check; a failure stops the run."""
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        raise SystemExit(f"check failed: {name}: {detail}")


def split_indices(splits, n_days):
    """Day indices of train / validation / test from the config's inclusive ranges, checked."""
    idx = {k: np.arange(splits[k][0], splits[k][1] + 1) for k in ("train", "validation", "test")}
    check_split(idx, splits["min_separation_days"], n_days)
    return idx


def check_split(idx, min_separation, n_days):
    """Check 9: sets inside the record, disjoint, and any two days of different sets at
    least ``min_separation`` apart."""
    names = list(idx)
    for k in names:
        if idx[k].min() < 0 or idx[k].max() >= n_days:
            raise ValueError(f"{k} days outside 0..{n_days - 1}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = int(np.abs(idx[a][:, None] - idx[b][None, :]).min())
            if gap < min_separation:
                raise ValueError(f"{a} and {b} come within {gap} days; need >= {min_separation}")


def floored_f(ff, f_min):
    """f with |f| floored at ``f_min``, sign kept, + at f == 0: the denominator of
    legoesm.ml.eke_unet.surface_vorticity_over_f."""
    ff = np.asarray(ff, np.float64)  # a float32 ff would round f_min to float32
    return np.where(ff < 0, np.minimum(ff, -f_min), np.maximum(ff, f_min))


def scored_cells(tmask, wet_coverage, dropped_columns, coverage_min):
    """S: surface wet, wet_coverage >= coverage_min, column not dropped."""
    s = (np.asarray(tmask) > 0) & (np.asarray(wet_coverage) >= coverage_min)
    s[:, list(dropped_columns)] = False
    return s


def weighted_mean_std(x, w):
    """Area-weighted mean and std, float64, two-pass."""
    x = np.asarray(x, np.float64)
    w = np.broadcast_to(np.asarray(w, np.float64), x.shape)
    sw = w.sum()
    mean = (w * x).sum() / sw
    return float(mean), float(np.sqrt((w * (x - mean) ** 2).sum() / sw))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_inputs(data, dropped_columns, coverage_min):
    """Read the four input files and run check 1 (read-back of the spec's section 4 table).

    ``vor`` is on disk as (y_c, x_c, t); it is transposed by name. Returns a dict of
    numpy arrays: vor, eke (t, y, x) float64; tmask (surface, bool), ff, area, lat, lon,
    wet_coverage (y, x); scored (bool).
    """
    d, ex = Path(data["dir"]), data["expect"]
    mesh = xr.open_dataset(d / data["mesh_file"])
    cov = xr.open_dataset(d / data["coverage_file"])
    dv = xr.open_dataset(d / data["vor_file"], decode_times=False)
    de = xr.open_dataset(d / data["eke_file"], decode_times=False)
    pv, pe = dv[data["vor_var"]], de[data["eke_var"]]
    check("1 dims", pv.dims == ("y_c", "x_c", "t") and pe.dims == ("t", "y_c", "x_c")
          and pv.sizes["y_c"] == pe.sizes["y_c"] == ex["ny"] and pv.sizes["x_c"] == pe.sizes["x_c"] == ex["nx"]
          and pv.sizes["t"] == pe.sizes["t"] == data["n_days"], f"vor {pv.dims} {pv.shape}, eke {pe.dims} {pe.shape}")
    check("1 dtypes", pv.dtype == np.float64 and pe.dtype == np.float64, f"{pv.dtype}, {pe.dtype}")
    t = np.arange(data["n_days"])
    check("1 time axes", np.array_equal(dv["t"].values, de["t"].values) and np.array_equal(dv["t"].values, t),
          f"both 0..{data['n_days'] - 1} step 1: {dv['t'].attrs.get('units')}")
    vor = pv.transpose("t", "y_c", "x_c").values
    eke = pe.values
    tmask = mesh["tmask"].isel(t=0, nav_lev=0).values > 0
    ff = mesh["ff_t"].isel(t=0).values
    area = (mesh["e1t"] * mesh["e2t"]).isel(t=0).values
    lat, lon = mesh["gphit"].isel(t=0).values, mesh["glamt"].isel(t=0).values
    wc = cov[data["coverage_var"]].values
    check("1 shapes of mesh fields", all(a.shape == (ex["ny"], ex["nx"]) for a in (tmask, ff, area, lat, lon, wc)),
          f"tmask {tmask.shape}")
    check("1 NaN", not np.isnan(vor).any() and not np.isnan(eke).any(), "0 NaN in vor and fine_ke")
    check("1 land is 0", not vor[:, ~tmask].any() and not eke[:, ~tmask].any(), "vor and fine_ke are 0 on land")
    check("1 wet cells", int(tmask.sum()) == ex["wet_cells"], f"{int(tmask.sum())} wet surface cells")
    check("1 land rows", not tmask[ex["land_rows"]].any(), f"rows {ex['land_rows']} are land")
    zero = np.where((ff == 0).any(1))[0].tolist()
    check("1 f = 0 rows", zero == ex["zero_f_rows"] == np.where((ff == 0).all(1))[0].tolist(), f"ff_t == 0 on rows {zero}")
    r0, r1 = ex["ring_wet_rows"]
    ring = [np.where(tmask[:, c])[0] for c in dropped_columns]
    check("1 ring columns", all(np.array_equal(r, np.arange(r0, r1 + 1)) for r in ring),
          f"columns {list(dropped_columns)} wet on rows {[(int(r.min()), int(r.max()), r.size) for r in ring]}")
    check("1 EKE > 0 on wet cells", bool((eke[:, tmask] > 0).all()) and bool(np.isfinite(eke[:, tmask]).all()),
          f"fine_ke wet min {eke[:, tmask].min():.4g}, max {eke[:, tmask].max():.4g}")
    low = tmask & (wc < coverage_min)
    check("1 low-coverage cells", int(low.sum()) == ex["low_coverage_wet_cells"]
          and not low[:, [c for c in range(ex["nx"]) if c not in dropped_columns]].any(),
          f"{int(low.sum())} wet cells with wet_coverage < {coverage_min}, all in columns {list(dropped_columns)}")
    scored = scored_cells(tmask, wc, dropped_columns, coverage_min)
    check("1 scored cells", int(scored.sum()) == ex["scored_cells"], f"{int(scored.sum())} scored cells")
    return dict(vor=vor, eke=eke, tmask=tmask, ff=ff, area=area, lat=lat, lon=lon, wet_coverage=wc, scored=scored)


def check_grid(tmask, ff, grid_cfg, rel_tol, zero_f_rows):
    """Check 2: the legoESM 1 deg DINO grid on which the network runs online has the file's
    surface land mask in every cell and its Coriolis parameter on wet cells outside row 99."""
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.experiments.dino import nemo_faithful_dino_domain

    set_policy(PrecisionPolicy.fp64())  # the NEMO-faithful domain cannot be built below fp64
    dom = nemo_faithful_dino_domain(grid_cfg["nemo_resolution"], grid_cfg["nemo_channel_rows"])
    land = np.asarray(dom.land_mask) > 0
    check("2 land mask", land.shape == tmask.shape and bool((land == tmask).all()),
          f"legoESM {grid_cfg['nemo_channel_rows']} surface land mask {land.shape}, wet {int(land.sum())}; "
          f"cells unequal to tmask[0, 0]: {int((land != tmask).sum())}")
    f = np.asarray(dom.geometry.grid_coriolis)
    sel = tmask.copy()
    sel[zero_f_rows] = False
    rel = np.abs(f[sel] - ff[sel]) / np.abs(ff[sel])
    check("2 Coriolis", f.shape == ff.shape and float(rel.max()) <= rel_tol,
          f"grid_coriolis vs ff_t on {int(sel.sum())} wet cells outside rows {zero_f_rows}: max relative difference {rel.max():.3g}")


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def replace_atomic(write, final):
    """``write(path)`` to a temporary name beside ``final`` (ending .npz), then rename."""
    final = Path(final)
    tmp = final.with_name(final.stem + ".tmp.npz")
    write(tmp)
    os.replace(tmp, final)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = load_config(args.config)
    data, out_dir = cfg["data"], Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from legoesm.ml import eke_unet
    from legoesm.ml.eke_unet import F_MIN, WRAP_COLUMNS, load_eke_unet, network_input, predict_eke, wall_fill_index

    where = Path(eke_unet.__file__).resolve()
    print(f"legoesm.ml.eke_unet.__file__ = {where}", flush=True)
    check("0 eke_unet is the worktree's", REPO in where.parents, f"repo {REPO}")
    print(f"F_MIN = {F_MIN}, WRAP_COLUMNS = {WRAP_COLUMNS}, config {args.config}", flush=True)

    idx = split_indices(cfg["splits"], data["n_days"])
    check("9 split", True, {k: (int(v[0]), int(v[-1]), v.size) for k, v in idx.items()}
          | {"min_separation_days": cfg["splits"]["min_separation_days"]})

    cols = cfg["dropped_columns"]
    r = read_inputs(data, cols, cfg["coverage_min"])
    tmask, scored, area, lat = r["tmask"], r["scored"], r["area"], r["lat"]
    check_grid(tmask, r["ff"], cfg["grid"], cfg["checks"]["grid_f_rel_tol"], data["expect"]["zero_f_rows"])

    # vorticity / f exactly as surface_vorticity_over_f forms it (float64, f floored at F_MIN);
    # predict_eke casts it to float32 first thing, so float32 storage loses nothing
    vof = np.where(tmask, r["vor"] / floored_f(r["ff"], F_MIN), 0.0)
    del r["vor"]
    vof32 = vof.astype(np.float32)

    # check 3: statistics on train days (the only days they use)
    tr = idx["train"]
    vof_tr = vof[tr]
    sel = scored & (np.abs(lat) >= cfg["stats_min_abs_lat_deg"])
    mean, std = weighted_mean_std(vof_tr[:, sel], area[sel])
    ck = cfg["checks"]
    check("3 statistics", abs(std - cfg["vor_norm"]["std"]) <= ck["stats_std_rel_tol"] * cfg["vor_norm"]["std"]
          and abs(mean) < ck["stats_abs_mean_max"],
          f"train days {tr[0]}..{tr[-1]}, |lat| >= {cfg['stats_min_abs_lat_deg']}, {int(sel.sum())} cells: "
          f"area-weighted mean {mean:.6f}, std {std:.6f}; config {cfg['vor_norm']}")
    m0, s0 = cfg["vor_norm"]["mean"], cfg["vor_norm"]["std"]
    for b in ck["clip_fraction"]["bands"]:
        s = scored & (np.abs(lat) >= b["min"]) & (np.abs(lat) < b["max"])
        frac = float((np.abs((vof_tr[:, s] - m0) / s0) > eke_unet.CLIP_SIGMA).mean())
        check(f"3 clipped fraction |lat| {b['min']}..{b['max']}", abs(frac - b["expect"]) <= ck["clip_fraction"]["tol"],
              f"{frac:.4f} of {int(s.sum())} cells x {tr.size} days (spec {b['expect']})")

    del vof_tr, vof
    # y and the D-c climatology
    ln_eke = np.log(np.where(scored, r["eke"], 1.0))
    y = np.where(scored, ln_eke, 0.0).astype(np.float32)
    clim = np.where(scored, ln_eke[tr].mean(0), 0.0)
    y_ring = np.where(tmask, np.log(np.where(tmask, r["eke"], 1.0)), 0.0)[:, :, cols].astype(np.float32)
    del ln_eke, r["eke"]

    # D-b: the base weights with the config's statistics; the network that builds X
    base = dict(np.load(cfg["base"]["npz"]))
    base["vor_norm"] = np.array([m0, s0], np.float64)
    db_path = out_dir / cfg["files"]["base_1deg_stats_npz"]
    replace_atomic(lambda t: np.savez(t, **base), db_path)
    net = load_eke_unet(db_path)
    check("0 vor_norm of the network that builds X", (net.vor_mean, net.vor_std) == (m0, s0), f"{(net.vor_mean, net.vor_std)}")

    import jax
    import jax.numpy as jnp
    maskf = tmask.astype(np.float64)
    fill = wall_fill_index(maskf, cfg["halo"], pad_ns=cfg["pad_ns"])
    mask_j = jnp.asarray(maskf)
    x0, h, w = network_input(net, jnp.asarray(vof32[0]), mask_j, fill, cfg["periodic_x"])
    ny, nx = tmask.shape
    check("4 shape and offsets", tuple(x0.shape) == tuple(data["expect"]["input_shape"])
          and (h, w) == (cfg["halo"], data["expect"]["wrap_columns"]) and WRAP_COLUMNS == w,
          f"network_input -> {tuple(x0.shape)}, {x0.dtype}, halo rows {h}, wrapped columns {w}; field at rows {h}:{h + ny}, columns {w}:{w + nx}")
    # eager, as ml_geometric_refresh_eke calls it online (a jitted version did not reproduce a fresh
    # eager call bit for bit on day 0, the first run of check 4)
    X = np.empty((data["n_days"], *x0.shape[:2]), np.float32)
    for d in range(data["n_days"]):
        X[d] = np.asarray(network_input(net, jnp.asarray(vof32[d]), mask_j, fill, cfg["periodic_x"])[0])[..., 0]
    check("4 X finite", bool(np.isfinite(X).all()), f"X {X.shape} {X.dtype}, min {X.min():.3f}, max {X.max():.3f}")

    # check 4: the network_input path is what predict_eke runs, and the stored X is that input
    for d in cfg["scoring"]["input_path_days"]:
        v = jnp.asarray(vof32[d])
        x, hh, ww = network_input(net, v, mask_j, fill, cfg["periodic_x"])
        via = jnp.where(mask_j > 0, jnp.exp(net(x)[hh:hh + ny, ww:ww + nx, 0]), 0.0)
        ref = predict_eke(net, v, mask_j, fill, periodic_x=cfg["periodic_x"])
        check(f"4 day {d}", bool(jnp.array_equal(via.astype(ref.dtype), ref)) and np.array_equal(X[d], np.asarray(x)[..., 0]),
              f"exp(net(network_input)) x mask == predict_eke (max |diff| {float(jnp.abs(via - ref).max()):.1e}); "
              f"stored X[{d}] == a fresh network_input")

    meta = {**git_state(), "created": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
            "config": cfg, "config_file": str(args.config), "numpy": np.__version__, "jax": jax.__version__,
            "input_sha256": {k: sha256(Path(data["dir"]) / data[k]) for k in
                             ("vor_file", "eke_file", "mesh_file", "coverage_file")},
            "base_npz_sha256": sha256(cfg["base"]["npz"]),
            "vor_norm_stats_recomputed": {"mean": mean, "std": std, "n_cells": int(sel.sum()), "days": [int(tr[0]), int(tr[-1])]}}
    arrays = dict(X=X, y=y, y_ring=y_ring, vor_over_f=vof32, scored=scored, tmask=tmask, area=area, lat=lat, lon=r["lon"],
                  idx_train=idx["train"], idx_validation=idx["validation"], idx_test=idx["test"], clim=clim,
                  crop=np.array([h, h + ny, w, w + nx]), ring_columns=np.array(cols), meta=np.array(json.dumps(meta, default=str)))
    final = out_dir / cfg["files"]["prepared"]
    replace_atomic(lambda t: np.savez(t, **arrays), final)
    print(f"wrote {final} ({final.stat().st_size / 1e9:.2f} GB) and {db_path}", flush=True)
    print(json.dumps({k: meta[k] for k in ("git_commit", "git_branch", "dirty", "input_sha256")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
