"""Fine-tune the DINO EKE U-Net (model_20260929-121036) on the 1 deg pairs (spec section 5).

Runs in the TensorFlow environment (``~/Python/AI4PEX/tensorflow_env``; no JAX, no
legoESM import), reading ``finetune_data_1deg.npz`` from
``scripts/data/prepare_eke_unet_finetune_data.py``. All values come from the config
(``scripts/experiment/dino/eke_unet_finetune_1deg.yaml``).

Modes::

    python finetune_eke_unet_keras.py --config CFG --self-check        # CPU, asserts, writes nothing
    python finetune_eke_unet_keras.py --config CFG --base-reference    # CPU, writes keras_base_on_X.npz (check 8)
    python finetune_eke_unet_keras.py --config CFG                     # training (GPU), writes the product

Training. The base network is fitted through a wrapper ``keras.Model`` whose ``call``
runs ``base(x, training=False)`` and crops the field out of the halo, so BatchNorm
keeps its moving statistics (as in the JAX port, which freezes them) and dropout is
off; the 31 trainable arrays (every conv kernel, BatchNorm scale and shift, the output
kernel) train. Loss: sum(m (y - yhat)^2) / sum(m) over a batch, ``m`` the scored cells,
carried in ``y_true`` as a second channel. Adam, constant learning rate, no EMA;
early stopping on ``val_loss`` with the best weights restored. The base ``.keras``
model (not the wrapper) is saved with a provenance record, as the converter
(``convert_eke_unet_keras_to_npz.py``) reads it: ``run_settings`` holds ``features``,
``target``, ``n_lags`` and ``norm_stats``.

Writes under ``out_dir`` (each under a temporary name, then renamed):
``model_<run_id>_ft1deg.keras``, ``history_<run_id>.json``, ``pred_test_keras.npy``
(log EKE of the reloaded saved model on the test days, for check A6).
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "scripts" / "experiment" / "dino" / "eke_unet_finetune_1deg.yaml"


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        raise SystemExit(f"check failed: {name}: {detail}")


def tf_imports(code_dir):
    sys.path.insert(0, str(Path(code_dir).expanduser()))
    import keras
    import model_components  # noqa: F401  registers cnn>Scenario, ReplicationPadding2D
    import tensorflow as tf
    import unet  # noqa: F401  registers unet>UNet
    return keras, tf


def make_wrapper(keras, base, crop):
    r0, r1, c0, c1 = (int(c) for c in crop)

    class CroppedUNet(keras.Model):
        """base(x, training=False), cropped to the field: BatchNorm and dropout stay in inference mode."""

        def __init__(self, base_model, **kw):
            super().__init__(**kw)
            self.base = base_model

        def call(self, x, training=False):
            return self.base(x, training=False)[:, r0:r1, c0:c1, :]

    return CroppedUNet(base)


def masked_mse(tf):
    def loss(y_true, y_pred):  # y_true[..., 0] target, y_true[..., 1] scored-cell mask
        m = y_true[..., 1:]
        return tf.reduce_sum(m * tf.square(y_true[..., :1] - y_pred)) / tf.reduce_sum(m)
    return loss


def load_prepared(cfg):
    z = np.load(Path(cfg["out_dir"]) / cfg["files"]["prepared"])
    return {k: z[k] for k in z.files}


def xy(d, idx):
    """Inputs (n, rows, cols, 1) and targets (n, ny, nx, 2) = (ln EKE, scored mask) for day indices ``idx``."""
    x = d["X"][idx][..., None]
    y = d["y"][idx][..., None]
    m = np.broadcast_to(d["scored"][None, :, :, None].astype(np.float32), y.shape)
    return x, np.concatenate([y, m], axis=-1)


def snapshot(base):
    return {v.path: v.numpy().copy() for v in base.variables}


def load_base(keras, cfg):
    keras.utils.set_random_seed(cfg["training"]["seed"])  # before the model is loaded
    return keras.saving.load_model(str(Path(cfg["base"]["keras"]).expanduser()), compile=False)


def compile_model(keras, tf, model, cfg):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"]), loss=masked_mse(tf))


def check_counts(base, cfg):
    e = cfg["base"]["expect"]
    n_tr = sum(int(np.prod(v.shape)) for v in base.trainable_variables)
    n_non = sum(int(np.prod(v.shape)) for v in base.non_trainable_variables)
    moving = [v for v in base.non_trainable_variables if "moving" in v.path]
    check("trainable / non-trainable", (len(base.trainable_variables), n_tr, n_non, len(moving)) ==
          (e["n_trainable_arrays"], e["n_trainable_values"], e["n_non_trainable_values"], e["n_bn_moving_arrays"]),
          f"{len(base.trainable_variables)} arrays, {n_tr} values trainable; {n_non} values non-trainable, {len(moving)} BatchNorm moving arrays")


def check_frozen_and_moved(base, before, n_trainable, label):
    after = snapshot(base)
    moving = [k for k in before if "moving" in k]
    same = [k for k in moving if np.array_equal(before[k], after[k])]
    check(f"{label}: moving statistics unchanged to the bit", len(same) == len(moving) == 20, f"{len(same)} of {len(moving)} arrays")
    tr = {v.path for v in base.trainable_variables}
    moved = [k for k in tr if not np.array_equal(before[k], after[k])]
    check(f"{label}: trainable arrays changed", len(moved) == len(tr) == n_trainable, f"{len(moved)} of {len(tr)} arrays")


def self_check(cfg):
    keras, tf = tf_imports(cfg["base"]["code_dir"])
    d = load_prepared(cfg)
    idx = d["idx_train"][::800][:4]
    x, y = xy(d, idx)
    base = load_base(keras, cfg)
    check_counts(base, cfg)
    model = make_wrapper(keras, base, d["crop"])
    compile_model(keras, tf, model, cfg)
    # the loss equals a numpy evaluation (float32 TF against float64 numpy: relative 1e-6)
    pred = model.predict(x, batch_size=len(idx), verbose=0)
    m = y[..., 1:].astype(np.float64)
    ref = float((m * (y[..., :1].astype(np.float64) - pred.astype(np.float64)) ** 2).sum() / m.sum())
    got = float(model.evaluate(x, y, batch_size=len(idx), verbose=0))
    check("loss == numpy", abs(got - ref) <= 1e-6 * max(1.0, abs(ref)), f"keras {got:.8f}, numpy {ref:.8f} on {len(idx)} days {idx.tolist()}")
    # changing targets outside S leaves the loss unchanged
    y2 = y.copy()
    out = y[..., 1:] == 0
    y2[..., :1][out] = 1.0e3
    got2 = float(model.evaluate(x, y2, batch_size=len(idx), verbose=0))
    check("targets outside S do not matter", got2 == got and bool(out.any()), f"loss {got2:.8f} with {int(out.sum())} unscored target values set to 1e3")
    # 2 steps: moving statistics unchanged to the bit, all trainable arrays changed
    before = snapshot(base)
    h = model.fit(x, y, batch_size=2, epochs=1, shuffle=False, verbose=0)
    check("2 steps ran", len(h.history["loss"]) == 1 and np.isfinite(h.history["loss"][0]), f"mean loss over the 2 steps {h.history['loss'][0]:.6f}")
    check_frozen_and_moved(base, before, cfg["base"]["expect"]["n_trainable_arrays"], "after 2 steps")
    print("self-check passed", flush=True)


def base_reference(cfg):
    """Check 8, first half: the Keras base model on the stored X of the check days."""
    keras, tf = tf_imports(cfg["base"]["code_dir"])
    d = load_prepared(cfg)
    s = cfg["scoring"]["keras_check_days"]
    days = s["start"] + s["step"] * np.arange(s["count"])
    check("check-8 days are test days", bool(np.isin(days, d["idx_test"]).all()), f"{days.tolist()}")
    base = load_base(keras, cfg)
    model = make_wrapper(keras, base, d["crop"])
    x, _ = xy(d, days)
    log_eke = model.predict(x, batch_size=cfg["training"]["batch_size"], verbose=0)[..., 0].astype(np.float32)
    out = Path(cfg["out_dir"]) / cfg["files"]["keras_base_reference"]
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, days=days, log_eke=log_eke)
    os.replace(tmp, out)
    print(f"wrote {out}: {log_eke.shape}, finite {bool(np.isfinite(log_eke).all())}", flush=True)


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def train(cfg):
    keras, tf = tf_imports(cfg["base"]["code_dir"])
    tcfg = cfg["training"]
    start = datetime.now()
    run_id = start.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg["out_dir"])
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"run_id {run_id}; tensorflow {tf.__version__}, keras {keras.__version__}; host {socket.gethostname()}; "
          f"SLURM_JOB_ID {os.environ.get('SLURM_JOB_ID')}; GPUs {[tf.config.experimental.get_device_details(g).get('device_name') for g in gpus]}", flush=True)
    check("GPU present", bool(gpus) or not tcfg["require_gpu"], f"{len(gpus)} GPU(s)")
    d = load_prepared(cfg)
    meta = json.loads(str(d["meta"]))
    xtr, ytr = xy(d, d["idx_train"])
    xva, yva = xy(d, d["idx_validation"])
    print(f"seed {tcfg['seed']}; train {xtr.shape} {ytr.shape}, validation {xva.shape}; scored cells {int(d['scored'].sum())}", flush=True)

    base = load_base(keras, cfg)
    check_counts(base, cfg)
    base_prov = base.provenance
    init = snapshot(base)
    model = make_wrapper(keras, base, d["crop"])
    compile_model(keras, tf, model, cfg)
    print("optimizer", json.dumps(model.optimizer.get_config(), default=str), flush=True)
    val0 = float(model.evaluate(xva, yva, batch_size=tcfg["batch_size"], verbose=0))
    print(f"validation loss before training (D-b on the validation days): {val0:.6f}", flush=True)

    class Guard(keras.callbacks.Callback):
        """Epoch timing; stops on a non-finite loss, or on val_loss above its pre-training value at epoch N (section 10)."""

        def __init__(self):
            super().__init__()
            self.seconds, self.problem, self._t = [], None, None

        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.time()

        def on_epoch_end(self, epoch, logs=None):
            self.seconds.append(time.time() - self._t)
            if not all(np.isfinite(logs[k]) for k in ("loss", "val_loss")):
                self.problem = f"non-finite loss at epoch {epoch + 1}: {logs}"
            elif epoch + 1 == tcfg["halt_if_val_above_start_at_epoch"] and logs["val_loss"] > val0:
                self.problem = f"val_loss {logs['val_loss']:.6f} at epoch {epoch + 1} above its value before training {val0:.6f}"
            if self.problem:
                print("STOP:", self.problem, flush=True)
                self.model.stop_training = True

    guard = Guard()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=tcfg["early_stopping_patience"],
                                       min_delta=tcfg["early_stopping_min_delta"], restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(xtr, ytr, batch_size=tcfg["batch_size"], epochs=tcfg["max_epochs"], shuffle=True,
                     validation_data=(xva, yva), callbacks=[es, guard], verbose=2)
    wall = time.time() - t0
    m = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    peak = None
    if gpus:
        peak = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    record = {"run_id": run_id, "val_loss_before_training": val0, "epochs_run": len(m["loss"]), "loss": m["loss"], "val_loss": m["val_loss"],
              "epoch_seconds": guard.seconds, "wall_seconds_fit": wall, "gpu_peak_bytes_tf": peak, "problem": guard.problem,
              "seed": tcfg["seed"], "steps_per_epoch": int(np.ceil(len(xtr) / tcfg["batch_size"]))}
    if guard.problem:
        path = out_dir / f"history_{run_id}.json"
        path.write_text(json.dumps(record, indent=1))
        raise SystemExit(f"stopped: {guard.problem}; curve in {path}")
    best = int(es.best_epoch) + 1  # 1-based
    stopped = es.stopped_epoch + 1 if es.stopped_epoch > 0 else None
    record |= {"best_epoch": best, "early_stopped_at_epoch": stopped, "train_loss_at_best": m["loss"][best - 1],
               "val_loss_at_best": m["val_loss"][best - 1]}
    print(f"epochs run {record['epochs_run']}, best epoch {best}, train loss {record['train_loss_at_best']:.6f}, "
          f"val loss {record['val_loss_at_best']:.6f}, fit wall {wall:.0f} s, GPU peak {peak}", flush=True)
    now = float(model.evaluate(xva, yva, batch_size=tcfg["batch_size"], verbose=0))
    check("best weights restored", abs(now - record["val_loss_at_best"]) <= 1e-3 * record["val_loss_at_best"], f"val loss now {now:.6f}, best epoch's {record['val_loss_at_best']:.6f}")
    check_frozen_and_moved(base, init, cfg["base"]["expect"]["n_trainable_arrays"], "after training")

    end = datetime.now()
    environment = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
                   "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "gpus": [tf.config.experimental.get_device_details(g).get("device_name") for g in gpus]}
    training_history = {"start_time": start.isoformat(timespec="seconds"), "end_time": end.isoformat(timespec="seconds"),
                        "epochs_completed": record["epochs_run"], "steps_per_epoch": record["steps_per_epoch"],
                        "validation_steps": int(np.ceil(len(xva) / tcfg["batch_size"])), "early_stopped_at_epoch": stopped,
                        "restored_best_weights": True, "weights_from_epoch": best, "metrics": m,
                        "val_loss_before_training": val0, "epoch_seconds": guard.seconds}
    run_settings = {"features": ["vor"], "target": ["fine_ke"], "n_lags": 0,
                    "norm_stats": {"vor": {"mean": cfg["vor_norm"]["mean"], "std": cfg["vor_norm"]["std"]}}, **cfg}
    base.provenance = {"run_id": run_id, "run_settings": run_settings, "base_model": base_prov, "data": meta,
                       "code_version": git_state(), "environment": environment, "training_history": training_history}
    final = out_dir / f"model_{run_id}_ft1deg.keras"
    tmp = out_dir / f"model_{run_id}_ft1deg.tmp.keras"
    base.save(str(tmp))
    # reload: provenance kept, output identical, moving statistics equal to the base's bit for bit
    base2 = keras.saving.load_model(str(tmp), compile=False)
    check("reload keeps provenance", base2.provenance is not None and base2.provenance["run_id"] == run_id, "assets/provenance.json")
    base_fresh = keras.saving.load_model(str(Path(cfg["base"]["keras"]).expanduser()), compile=False)
    mv0, mv1 = ({v.path: v.numpy() for v in b.non_trainable_variables if "moving" in v.path} for b in (base_fresh, base2))
    check("saved moving statistics == the base file's, bit for bit", mv0.keys() == mv1.keys() and len(mv0) == 20
          and all(np.array_equal(mv0[k], mv1[k]) for k in mv0), f"{len(mv0)} arrays")
    model2 = make_wrapper(keras, base2, d["crop"])
    probe = xva[:4]
    diff = float(np.abs(model.predict(probe, verbose=0) - model2.predict(probe, verbose=0)).max())
    check("reloaded model == trained model", diff == 0.0, f"max |diff| of log EKE on 4 validation days {diff:.3g}")
    os.replace(tmp, final)

    # the test days are read only here, for A6
    xte, _ = xy(d, d["idx_test"])
    pred = model2.predict(xte, batch_size=tcfg["batch_size"], verbose=0)[..., 0].astype(np.float32)
    check("test predictions finite", bool(np.isfinite(pred).all()), f"{pred.shape} log EKE, min {pred.min():.3f}, max {pred.max():.3f}")
    ptmp = out_dir / "pred_test_keras.tmp.npy"
    np.save(ptmp, pred)
    os.replace(ptmp, out_dir / cfg["files"]["pred_test_keras"])
    htmp = out_dir / f"history_{run_id}.json.tmp"
    htmp.write_text(json.dumps(record, indent=1))
    os.replace(htmp, out_dir / f"history_{run_id}.json")
    print(f"wrote {final}\nwrote {out_dir / cfg['files']['pred_test_keras']}\nwrote {out_dir / f'history_{run_id}.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--base-reference", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.self_check:
        self_check(cfg)
    elif args.base_reference:
        base_reference(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
