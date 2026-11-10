#!/usr/bin/env python
"""jax.grad vs finite differences for the GEOMETRIC eddy efficiency alpha,
through the DINO driver's own build and step (run_dino.build_run / advance).

    JAX_ENABLE_X64=1 python scripts/validate/dino_alpha_gradient_check.py \
        --steps 2 --config scripts/experiment/dino/nemo_faithful_kamm_mlf_r4.yaml

Flags other than --steps / --eps-rel / --seed go to run_dino's own parser, so
the model, state and forcing are those of a run with the same flags.

L(alpha) = sum(w_T * T + w_S * S) over the state after --steps steps, with
fixed random weights: a projection, because a norm has zero derivative
wherever the response is zero. FD is printed at eps and eps/10 so its own
truncation/round-off can be judged next to the AD value.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def _load_run_dino():
    spec = importlib.util.spec_from_file_location(
        "run_dino", _REPO / "scripts" / "run" / "run_dino.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    return rd


def alpha_gradient_check(run_dino_argv, n_steps=2, eps_rel=1.0e-4, seed=0,
                         n_devices=1):
    """Return a dict with alpha, L, AD dL/dalpha, FD at eps and eps/10, and
    wall times. ``run_dino_argv`` is parsed by run_dino's own parser. Runs
    under the fp64 precision policy (restored afterwards): a float32 state
    cannot resolve the FD difference. The NEMO cards force fp64 anyway.
    ``n_devices > 1`` splits the initial state along longitude over that many
    devices and lets XLA partition the step (no model change)."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prior = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        return _check(run_dino_argv, n_steps, eps_rel, seed, n_devices)
    finally:
        set_policy(prior)


def _split_along_longitude(state, n_devices):
    """Shard every leaf whose axis 1 is the longitude axis of T (n_lat, n_lon,
    nz); replicate the rest. The repo's lat-lon sharders split latitude, and
    the 1/4 deg DINO mesh has 797 rows (prime)."""
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    n_lon = state.T.data.shape[1]
    if n_lon % n_devices:
        raise SystemExit(f"{n_lon} longitude columns do not split evenly "
                         f"over {n_devices} devices")
    mesh = Mesh(np.array(jax.devices()[:n_devices]), ("lon",))
    lon, rep = NamedSharding(mesh, P(None, "lon")), NamedSharding(mesh, P())
    return jax.tree_util.tree_map(
        lambda a: jax.device_put(
            a, lon if a.ndim >= 2 and a.shape[1] == n_lon else rep)
        if isinstance(a, jax.Array) else a, state)


def _check(run_dino_argv, n_steps, eps_rel, seed, n_devices=1):
    import jax
    import jax.numpy as jnp

    if n_devices > len(jax.devices()):
        raise SystemExit(f"--n-devices {n_devices} but JAX sees "
                         f"{len(jax.devices())}: {jax.devices()}")
    rd = _load_run_dino()
    saved_argv = sys.argv
    sys.argv = ["run_dino", *run_dino_argv]
    try:
        args = rd._parse_args()
    finally:
        sys.argv = saved_argv
    run = rd.build_run(args)
    gm = run.model.config.gm_redi
    if gm is None or gm.eke is None or gm.eke.closure != "geometric":
        raise SystemExit("needs the GEOMETRIC closure (--gm-kappa-scheme "
                         "geometric, or a card that selects it)")
    alpha0 = float(gm.eke.geometric.alpha)
    if n_devices > 1:
        run.state = _split_along_longitude(run.state, n_devices)

    rng = np.random.default_rng(seed)
    w_T = jnp.asarray(rng.standard_normal(run.state.T.data.shape))
    w_S = jnp.asarray(rng.standard_normal(run.state.S.data.shape))
    # Each call gets a fresh model copy (the step's jit is keyed on the model
    # object); keep every copy alive so no id is reused by a later copy.
    _models = []

    def loss(alpha):
        m = copy.copy(run.model)
        g = m.config.gm_redi
        m.config = m.config._replace(gm_redi=g._replace(eke=g.eke._replace(
            geometric=g.eke.geometric._replace(alpha=alpha))))
        _models.append(m)
        state = run.state
        for k in range(n_steps):
            state = rd.advance(run, state, k, model=m)
        return jnp.sum(w_T * state.T.data) + jnp.sum(w_S * state.S.data)

    t0 = time.time()
    L, dL = jax.value_and_grad(loss)(jnp.asarray(alpha0))
    dL = float(jax.block_until_ready(dL))
    t_grad = time.time() - t0
    # Peak after the gradient, before the (smaller) FD forward runs; None on CPU.
    peak = [(d.memory_stats() or {}).get("peak_bytes_in_use")
            for d in jax.devices()[:n_devices]]

    # alpha as a jit input: one compile serves all four FD evaluations.
    loss_jit = jax.jit(loss)
    t0 = time.time()
    fd = {}
    for e in (eps_rel * alpha0, 0.1 * eps_rel * alpha0):
        fd[e] = (float(loss_jit(alpha0 + e))
                 - float(loss_jit(alpha0 - e))) / (2 * e)
    t_fd = time.time() - t0
    return {"alpha": alpha0, "L": float(L), "ad": dL, "fd": fd,
            "steps": n_steps, "t_grad_s": t_grad, "t_fd_s": t_fd,
            "dt": run.dt, "grid": run.grid_desc, "n_devices": n_devices,
            "peak_bytes": peak}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--eps-rel", type=float, default=1.0e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-devices", type=int, default=1)
    own, rest = p.parse_known_args()

    import jax
    sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short",
                          "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(_REPO), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    print(f"commit {sha}{' (+uncommitted changes)' if dirty else ''}  "
          f"jax {jax.__version__}  x64={jax.config.x64_enabled}  "
          f"devices={jax.devices()}")
    print(f"argv: {sys.argv[1:]}")

    r = alpha_gradient_check(rest, own.steps, own.eps_rel, own.seed,
                             own.n_devices)
    print(f"{r['grid']}, dt={r['dt']} s, {r['steps']} steps, "
          f"alpha={r['alpha']}, {r['n_devices']} device(s)")
    print("peak memory after grad per device: "
          + ", ".join("n/a" if b is None else f"{b / 2**30:.2f} GiB"
                      for b in r["peak_bytes"]))
    print(f"L = {r['L']:.17e}")
    print(f"AD dL/dalpha = {r['ad']:.10e}   ({r['t_grad_s']:.1f} s)")
    for e, v in r["fd"].items():
        rel = abs(r["ad"] - v) / max(abs(v), 1e-300)
        print(f"FD eps={e:.3e}: {v:.10e}   |AD-FD|/|FD| = {rel:.3e}")
    print(f"FD wall time {r['t_fd_s']:.1f} s (1 compile + 4 forward runs)")


if __name__ == "__main__":
    main()
