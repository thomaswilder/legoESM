"""Score EKE U-Net weights at 1 deg (spec section 6; checks 5-8 and 11 of section 7).

Runs in the legoESM venv on CPU, through the JAX port (``predict_eke``,
``periodic_x`` and the halo from the config) on the prepared data of
``scripts/data/prepare_eke_unet_finetune_data.py``. All sums in float64.
``y = ln(EKE_true)``, ``p = ln(EKE_pred)``, scored cells S, days from the config.

Two modes (``JAX_PLATFORMS=cpu``)::

    # before training: checks 5, 6, 7 (D-a, D-b on the 138 repro days, D-c on the test
    # days) and, with --keras-base, check 8
    python scripts/validate/eke_unet_1deg_skill.py --config CFG --mode reproduce \
        --keras-base OUT/keras_base_on_X.npz --out OUT/metrics_reproduce.json

    # after training: the test days for any number of weights files (D-c is always
    # added); --ft names the fine-tuned one, which gets the acceptance table A1-A6
    # and check 11
    python scripts/validate/eke_unet_1deg_skill.py --config CFG --mode score \
        --npz D-a=base.npz D-b=OUT/base_1degstats.npz ft=ft.npz --ft ft \
        --keras-pred OUT/pred_test_keras.npy --history OUT/history_<run_id>.json \
        --out OUT/metrics.json

A failed check of section 7 stops the run; an acceptance criterion that fails is
reported, not hidden.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "scripts" / "experiment" / "dino" / "eke_unet_finetune_1deg.yaml"
METRICS = ("r2", "bias", "exp_bias", "mean_ratio", "mean_ratio_area", "low10", "anomaly_corr", "corr_log", "n")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        raise SystemExit(f"check failed: {name}: {detail}")


def day_list(spec):
    return np.arange(spec["start"], spec["stop"], spec["step"])


def band_mask(abs_lat, lo, hi):
    return (abs_lat >= lo) & (abs_lat < hi)


def box_mask(lon, lat, box):
    return (lon >= box["lon"][0]) & (lon <= box["lon"][1]) & (lat >= box["lat"][0]) & (lat <= box["lat"][1])


def score(y, p, area, c, low10_factor):
    """Section 6 scores of log-EKE prediction ``p`` against ``y`` for (days, cells) arrays.

    ``area`` (cells,) and the climatology ``c`` (cells,) broadcast over days; ``c=None``
    leaves the anomaly correlation undefined. Everything is accumulated in float64.
    """
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    w = np.broadcast_to(np.asarray(area, np.float64), y.shape)
    ybar = y.mean()
    ey, ep = np.exp(y), np.exp(p)
    out = {"n": int(y.size),
           "r2": float(1.0 - ((y - p) ** 2).sum() / ((y - ybar) ** 2).sum()),
           "bias": float((p - y).mean()),
           "mean_ratio": float(ep.mean() / ey.mean()),
           "mean_ratio_area": float((w * ep).sum() / (w * ey).sum()),
           "low10": float((p < y - np.log(low10_factor)).mean()),
           "corr_log": float(np.corrcoef(y.ravel(), p.ravel())[0, 1]) if p.std() > 0 else None}
    out["exp_bias"] = float(np.exp(out["bias"]))
    out["anomaly_corr"] = None
    if c is not None:
        a, b = p - np.asarray(c, np.float64), y - np.asarray(c, np.float64)
        if a.std() > 0:
            out["anomaly_corr"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return out


def subsets(cfg, d):
    """Cell selections of section 6: name -> boolean (ny, nx) over scored cells, plus the
    ring columns (wet cells of the dropped columns, scored separately)."""
    al = np.abs(d["lat"])
    sets = {b["name"]: d["scored"] & band_mask(al, b["min"], b["max"]) for b in cfg["scoring"]["bands"]}
    sets |= {k: d["scored"] & box_mask(d["lon"], d["lat"], v) for k, v in cfg["scoring"]["boxes"].items()}
    return sets


def score_all(cfg, d, days, pred):
    """All subsets for ``pred`` (days, ny, nx) EKE (None = the D-c climatology)."""
    f = cfg["scoring"]["low10_factor"]
    y_all = d["y"][days].astype(np.float64)
    p_all = np.broadcast_to(d["clim"], y_all.shape) if pred is None else np.log(np.where(d["tmask"], pred, 1.0).astype(np.float64))
    out = {}
    for name, sel in subsets(cfg, d).items():
        out[name] = score(y_all[:, sel], p_all[:, sel], d["area"][sel], d["clim"][sel], f)
    cols = list(d["ring_columns"])
    ring = d["tmask"][:, cols]
    if pred is None:
        out["ring_columns"] = None  # the climatology is defined on scored cells only
    else:
        y_r = d["y_ring"][days].astype(np.float64)[:, ring]
        p_r = p_all[:, :, cols][:, ring]
        out["ring_columns"] = score(y_r, p_r, d["area"][:, cols][ring], None, f)
    return out


def predict_days(net, d, days, cfg):
    """EKE (days, ny, nx) float32 of ``net`` on the prepared vorticity/f, through predict_eke."""
    import jax
    import jax.numpy as jnp
    from legoesm.ml.eke_unet import predict_eke, wall_fill_index

    mask = jnp.asarray(d["tmask"].astype(np.float64))
    fill = wall_fill_index(np.asarray(mask), cfg["halo"], pad_ns=cfg["pad_ns"])
    f = jax.jit(lambda v: predict_eke(net, v, mask, fill, periodic_x=cfg["periodic_x"]))
    pred = np.stack([np.asarray(f(jnp.asarray(d["vor_over_f"][k]))) for k in days])
    wet = np.broadcast_to(d["tmask"], pred.shape)
    check("pred finite", bool(np.isfinite(pred).all()) and bool((pred[wet] > 0).all()),
          f"{pred.shape} {pred.dtype}, wet min {pred[wet].min():.3g}, max {pred[wet].max():.3g}")
    return pred


def load_prepared(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files if k not in ("X",)}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(name, got, expect, abs_tol, rel_tol=None):
    """Check a score against the planning number; ratios also pass on the relative tolerance."""
    ok = abs(got - expect) <= abs_tol or (rel_tol is not None and abs(got - expect) <= rel_tol * abs(expect))
    check(name, ok, f"{got:.4f} vs planning {expect} (|diff| {abs(got - expect):.2g}, abs tol {abs_tol}"
                    + (f", rel tol {rel_tol}" if rel_tol else "") + ")")


def load_net(path, cfg, expect_norm=None):
    from legoesm.ml.eke_unet import load_eke_unet
    net = load_eke_unet(path)
    if expect_norm is not None:
        check(f"vor_norm of {Path(path).name}", (net.vor_mean, net.vor_std) == tuple(expect_norm), f"{(net.vor_mean, net.vor_std)}")
    return net


def mode_reproduce(cfg, d, args):
    sc, ck = cfg["scoring"], cfg["checks"]
    repro, test = day_list(sc["repro_days"]), day_list(sc["test_days"])
    out_dir = Path(cfg["out_dir"])
    res = {"repro_days": [int(repro[0]), int(repro[-1]), int(repro.size)]}
    preds = {}
    for label, path in (("D-a", cfg["base"]["npz"]), ("D-b", out_dir / cfg["files"]["base_1deg_stats_npz"])):
        net = load_net(path, cfg)
        print(f"{label}: weights {path}, vor_norm {(net.vor_mean, net.vor_std)}", flush=True)
        preds[label] = predict_days(net, d, repro, cfg)
        res[label] = score_all(cfg, d, repro, preds[label])
    a, e = res["D-a"], ck["D_a"]
    for band in ("all", "abs_lat_ge_20", "abs_lat_lt_20"):
        for k, v in e[band].items():
            key = k
            compare(f"5 D-a {band} {key}", a[band][key], v, e["abs_tol"], e["rel_tol"] if key.startswith("mean_ratio") else None)
    b, e = res["D-b"], ck["D_b"]
    for band in ("all", "abs_lat_ge_20"):
        for k, v in e[band].items():
            compare(f"6 D-b {band} {k}", b[band][k], v, e["tol"])
    res["D-c"] = score_all(cfg, d, test, None)
    lo, hi = ck["D_c_r2_range"]
    check("7 D-c R2", lo <= res["D-c"]["all"]["r2"] <= hi, f"{res['D-c']['all']['r2']:.4f} on test days {test[0]}..{test[-1]}, range {lo}..{hi}")
    if args.keras_base:
        ref = np.load(args.keras_base)
        days = ref["days"]
        pred = predict_days(load_net(out_dir / cfg["files"]["base_1deg_stats_npz"], cfg), d, days, cfg)
        wet = d["tmask"]
        diff = np.abs(np.log(pred[:, wet].astype(np.float64)) - ref["log_eke"][:, wet].astype(np.float64))
        res["check8_max_abs_diff"] = float(diff.max())
        check("8 Keras base on stored X vs JAX D-b", float(diff.max()) <= ck["keras_vs_jax_max_abs_log_eke"],
              f"{days.size} test days {days[0]}..{days[-1]}, {int(wet.sum())} wet cells: max |diff| of log EKE {diff.max():.3g} "
              f"(limit {ck['keras_vs_jax_max_abs_log_eke']})")
    return res


def acceptance(cfg, res, ft, db, extra):
    """A1-A5 (A6 from ``extra``) for the fine-tuned label; each with its number."""
    a, f, g = cfg["acceptance"], res[ft], res[db]
    bands = ("all", "abs_lat_ge_20", "abs_lat_lt_20")
    out = {}
    out["A1"] = {"pass": bool(f["all"]["r2"] >= a["A1"]["r2_min"] and f["abs_lat_ge_20"]["r2"] >= a["A1"]["r2_min"]
                              and f["abs_lat_lt_20"]["r2"] > a["A1"]["r2_min_lt20_exclusive"]),
                 "r2": {b: f[b]["r2"] for b in bands}}
    out["A2"] = {"pass": all(abs(f[b]["bias"]) <= a["A2"]["abs_bias_max"][b] for b in bands),
                 "bias": {b: f[b]["bias"] for b in bands}}
    out["A3"] = {"pass": all(a["A3"]["ratio_min"] <= f[b]["mean_ratio"] <= a["A3"]["ratio_max"] for b in a["A3"]["bands"]),
                 "mean_ratio": {b: f[b]["mean_ratio"] for b in bands}, "mean_ratio_area": {b: f[b]["mean_ratio_area"] for b in bands}}
    out["A4"] = {"pass": bool(f["all"]["low10"] <= a["A4"]["low10_max"]), "low10": f["all"]["low10"]}
    out["A5"] = {"pass": bool(f["all"]["r2"] - g["all"]["r2"] > a["A5"]["r2_margin_over_Db_min_exclusive"]),
                 "r2_ft": f["all"]["r2"], "r2_Db": g["all"]["r2"], "margin": f["all"]["r2"] - g["all"]["r2"]}
    if "keras_vs_jax_max_abs_log_eke" in extra:
        m = extra["keras_vs_jax_max_abs_log_eke"]
        out["A6"] = {"pass": bool(m <= a["A6"]["max_abs_log_eke_diff"]), "max_abs_diff": m}
    return out


def mode_score(cfg, d, args):
    sc, ck = cfg["scoring"], cfg["checks"]
    test = day_list(sc["test_days"])
    paths = dict(s.split("=", 1) for s in args.npz)
    res, extra = {"days": [int(test[0]), int(test[-1]), int(test.size)], "weights": {}}, {}
    preds = {}
    for label, path in paths.items():
        net = load_net(path, cfg)
        res["weights"][label] = {"path": str(path), "sha256": sha256(path), "vor_norm": [net.vor_mean, net.vor_std]}
        print(f"{label}: {path}, vor_norm {(net.vor_mean, net.vor_std)}", flush=True)
        preds[label] = predict_days(net, d, test, cfg)
        res[label] = score_all(cfg, d, test, preds[label])
    res["D-c"] = score_all(cfg, d, test, None)
    if args.ft:
        ft = args.ft
        w = np.load(paths[ft])
        for k in w.files:  # check 11: moving statistics are the base's, bit for bit
            if k.endswith(("_mean", "_var")):
                b = np.load(cfg["base"]["npz"])[k]
                check(f"11 moving statistic {k}", np.array_equal(w[k], b), "equal to the base's")
        check("11 vor_norm of the fine-tuned file", tuple(w["vor_norm"]) == (cfg["vor_norm"]["mean"], cfg["vor_norm"]["std"]),
              f"{tuple(w['vor_norm'])} == config {cfg['vor_norm']}")
        if args.keras_pred:
            kp = np.load(args.keras_pred)
            check("A6 shape", kp.shape == preds[ft].shape, f"{kp.shape}")
            wet = d["tmask"]
            diff = np.abs(np.log(preds[ft][:, wet].astype(np.float64)) - kp[:, wet].astype(np.float64))
            extra["keras_vs_jax_max_abs_log_eke"] = float(diff.max())
            print(f"A6: max |log EKE JAX - Keras| over {test.size} test days, {int(wet.sum())} wet cells = {diff.max():.3g}", flush=True)
        if args.history:
            h = json.load(open(args.history))
            val = day_list({"start": int(d["idx_validation"][0]), "stop": int(d["idx_validation"][-1]) + 1, "step": 1})
            pv = predict_days(load_net(paths[ft], cfg), d, val, cfg)
            s = d["scored"]
            mse = float(((d["y"][val].astype(np.float64)[:, s] - np.log(pv[:, s].astype(np.float64))) ** 2).mean())
            best = h["val_loss"][h["best_epoch"] - 1]
            extra["val_mse_jax"], extra["val_loss_keras_best"] = mse, best
            check("11 JAX masked MSE on validation days vs best Keras val_loss",
                  abs(mse - best) <= ck["val_loss_rel_tol"] * best, f"JAX {mse:.6f}, Keras epoch {h['best_epoch']} {best:.6f}")
        res["acceptance"] = acceptance(cfg, res, ft, args.db, extra)
        res["extra"] = extra
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--mode", required=True, choices=("reproduce", "score"))
    p.add_argument("--data", help="prepared npz (default: out_dir/files.prepared of the config)")
    p.add_argument("--npz", nargs="*", default=[], help="LABEL=path weights files (score mode)")
    p.add_argument("--ft", help="label of the fine-tuned weights (score mode)")
    p.add_argument("--db", default="D-b", help="label of the D-b weights (A5)")
    p.add_argument("--keras-base", help="keras_base_on_X.npz (check 8)")
    p.add_argument("--keras-pred", help="pred_test_keras.npy of the fine-tuned model (A6)")
    p.add_argument("--history", help="history json of the training (check 11)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    from legoesm.ml import eke_unet
    where = Path(eke_unet.__file__).resolve()
    print(f"legoesm.ml.eke_unet.__file__ = {where}", flush=True)
    check("0 eke_unet is the worktree's", REPO in where.parents, f"repo {REPO}")
    data_path = Path(args.data or Path(cfg["out_dir"]) / cfg["files"]["prepared"])
    d = load_prepared(data_path)
    res = mode_reproduce(cfg, d, args) if args.mode == "reproduce" else mode_score(cfg, d, args)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    res["meta"] = {"created": datetime.now().isoformat(timespec="seconds"), "mode": args.mode, "git_commit": git("rev-parse", "HEAD"),
                   "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "config_file": str(args.config), "config": cfg,
                   "prepared": str(data_path), "prepared_sha256": sha256(data_path), "prepared_meta": d["meta"]}
    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out)
    print(f"wrote {out}")
    if "acceptance" in res:
        for k, v in res["acceptance"].items():
            print(f"{k}: {'PASS' if v['pass'] else 'FAIL'}  {json.dumps({a: b for a, b in v.items() if a != 'pass'})}")


if __name__ == "__main__":
    sys.exit(main())
