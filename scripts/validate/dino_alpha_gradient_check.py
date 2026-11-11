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
                         chunk=0, seed_carry=False,
                         compile_only=False, unet_index=None):
    """Return a dict with alpha, L, AD dL/dalpha, FD at eps and eps/10, and
    wall times. ``run_dino_argv`` is parsed by run_dino's own parser. Runs
    under the fp64 precision policy (restored afterwards): a float32 state
    cannot resolve the FD difference. The NEMO cards force fp64 anyway.
    ``chunk = K > 0`` runs the window as nested checkpointed scans: backprop
    keeps one state per K-step chunk, then K states and one step's
    intermediates while it recomputes a chunk. Both scans start from the
    model's seeded scan carry (``seed_carry`` gives the plain step loop the
    same start, for comparison). ``compile_only`` compiles the jitted
    value-and-gradient and returns XLA's memory estimate without running it.

    ``unet_index`` (a card with gm_kappa_scheme ml_geometric): differentiate
    w.r.t. that element of the U-Net's output kernel instead of alpha, with
    the network cast to float64. The U-Net runs at each model-day start: at
    every chunk boundary (``chunk`` must then be one day of steps) or, in the
    plain loop, at every step that starts a day."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prior = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        return _check(run_dino_argv, n_steps, eps_rel, seed, chunk,
                      seed_carry or chunk > 0, compile_only, unet_index)
    finally:
        set_policy(prior)


def _seeded_state(run):
    """``seed_scan_carry`` with the step kwargs ``run_dino.advance`` passes
    (forcing can change a carry leaf's dtype)."""
    kw = {}
    if getattr(run.cfg, "wind_through_step", False):
        kw["surface_forcing"] = run.sf_step
    if run.forcing is not None and run.sf_placement == "leapfrog_rhs":
        _, kw["external_tracer_rate"] = run.apply_forcing(
            run.state, run.forcing, run.z, run.cfg, run.dt,
            t_seconds=run.dt, return_rate=True)
    return run.model.seed_scan_carry(run.state, run.dt, **kw)


def _window(rd, run, state, model, n_steps, chunk, net=None):
    """``n_steps`` driver steps from ``state``. With ``net`` (ML-GEOMETRIC)
    the U-Net refreshes state.eke at each model-day start: per chunk (a chunk
    is one day) or, in the plain loop, at every step k that starts a day."""
    import jax.numpy as jnp

    if chunk:
        from legoesm.timestepping.scan_loop import integrate_scan_generic

        def one_step(c):
            return rd.advance(run, c[0], c[1], model=model), c[1] + 1

        def one_chunk(c):
            if net is not None:
                c = (run.refresh_eke(c[0], net), c[1])
            return integrate_scan_generic(
                c, one_step, chunk, checkpoint_interval=1,
                return_trajectory=False)[0]

        (state, _), _ = integrate_scan_generic(
            (state, jnp.asarray(0)), one_chunk, n_steps // chunk,
            checkpoint_interval=1, return_trajectory=False)
        return state
    for k in range(n_steps):
        if net is not None and k % run.steps_per_day == 0:
            state = run.refresh_eke(state, net)
        state = rd.advance(run, state, k, model=model)
    return state


def _check(run_dino_argv, n_steps, eps_rel, seed, chunk=0,
           seed_carry=False, compile_only=False, unet_index=None):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    if chunk < 0 or (chunk and n_steps % chunk):
        raise SystemExit(f"--chunk {chunk} must be 0 or divide --steps "
                         f"{n_steps}")
    rd = _load_run_dino()
    saved_argv = sys.argv
    sys.argv = ["run_dino", *run_dino_argv]
    try:
        args = rd._parse_args()
    finally:
        sys.argv = saved_argv
    run = rd.build_run(args)
    gm = run.model.config.gm_redi
    if unet_index is None:
        if gm is None or gm.eke is None or gm.eke.closure != "geometric":
            raise SystemExit("needs the GEOMETRIC closure (--gm-kappa-scheme "
                             "geometric, or a card that selects it)")
        x0 = float(gm.eke.geometric.alpha)
    else:
        if run.refresh_eke is None:
            raise SystemExit("unet_index needs --gm-kappa-scheme ml_geometric")
        if chunk and chunk != run.steps_per_day:
            raise SystemExit(f"--chunk {chunk} must be one model day "
                             f"({run.steps_per_day} steps) for ml_geometric")
        net64 = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, jnp.float64), run.net)
        x0 = float(net64.out_kernel[unet_index])
    if seed_carry:
        run.state = _seeded_state(run)

    rng = np.random.default_rng(seed)
    w_T = jnp.asarray(rng.standard_normal(run.state.T.data.shape))
    w_S = jnp.asarray(rng.standard_normal(run.state.S.data.shape))
    # Each call gets a fresh model copy (the step's jit is keyed on the model
    # object); keep every copy alive so no id is reused by a later copy.
    _models = []

    def loss(x):
        m, net = copy.copy(run.model), None
        _models.append(m)
        if unet_index is None:
            g = m.config.gm_redi
            m.config = m.config._replace(gm_redi=g._replace(eke=g.eke._replace(
                geometric=g.eke.geometric._replace(alpha=x))))
        else:
            net = eqx.tree_at(lambda t: t.out_kernel, net64,
                              net64.out_kernel.at[unet_index].set(x))
        state = _window(rd, run, run.state, m, n_steps, chunk, net)
        return jnp.sum(w_T * state.T.data) + jnp.sum(w_S * state.S.data)

    if compile_only:
        t0 = time.time()
        m = jax.jit(jax.value_and_grad(loss)).lower(
            jnp.asarray(x0)).compile().memory_analysis()
        return {"alpha": x0, "steps": n_steps, "chunk": chunk,
                "dt": run.dt, "grid": run.grid_desc,
                "t_compile_s": time.time() - t0,
                "memory": {k: getattr(m, k) for k in dir(m)
                           if k.endswith("_in_bytes")
                           and not k.startswith("host_")}}

    t0 = time.time()
    # Jitted so XLA plans the whole backward. Jitting does not shrink a full
    # 1/4 deg step's backward (~36 GiB from step 2 on); 1-step figures are low
    # because step 1 starts from a constant state (jobs 66693365, 66708115).
    L, dL = jax.jit(jax.value_and_grad(loss))(jnp.asarray(x0))
    dL = float(jax.block_until_ready(dL))
    t_grad = time.time() - t0
    # Peak after the gradient, before the (smaller) FD forward runs; None on CPU.
    peak = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")

    # alpha as a jit input: one compile serves all four FD evaluations.
    loss_jit = jax.jit(loss)
    t0 = time.time()
    fd = {}
    for e in (eps_rel * abs(x0), 0.1 * eps_rel * abs(x0)):
        fd[e] = (float(loss_jit(x0 + e))
                 - float(loss_jit(x0 - e))) / (2 * e)
    t_fd = time.time() - t0
    return {"alpha": x0, "L": float(L), "ad": dL, "fd": fd,
            "steps": n_steps, "t_grad_s": t_grad, "t_fd_s": t_fd,
            "dt": run.dt, "grid": run.grid_desc, "peak_bytes": peak,
            "chunk": chunk}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--eps-rel", type=float, default=1.0e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=0,
                   help="checkpointed chunk length in steps (0 = plain loop)")
    p.add_argument("--compile-only", action="store_true",
                   help="print XLA's memory estimate for the jitted gradient")
    p.add_argument("--unet-index", default=None,
                   help="ml_geometric cards: differentiate w.r.t. this "
                        "element of the U-Net output kernel, e.g. 0,0,0,0")
    own, rest = p.parse_known_args()
    unet_index = (None if own.unet_index is None
                  else tuple(int(i) for i in own.unet_index.split(",")))

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
                             own.chunk,
                             compile_only=own.compile_only,
                             unet_index=unet_index)
    print(f"{r['grid']}, dt={r['dt']} s, {r['steps']} steps, "
          f"{'alpha' if unet_index is None else f'unet {unet_index}'}"
          f"={r['alpha']}, "
          f"chunk={r['chunk'] or 'none (plain loop)'}")
    if own.compile_only:
        for k, v in r["memory"].items():
            print(f"{k}: {v / 2**30:.2f} GiB")
        print(f"compile time {r['t_compile_s']:.1f} s")
        return
    b = r["peak_bytes"]
    print("peak memory after grad: "
          + ("n/a" if b is None else f"{b / 2**30:.2f} GiB"))
    print(f"L = {r['L']:.17e}")
    print(f"AD dL/dparam = {r['ad']:.10e}   ({r['t_grad_s']:.1f} s)")
    for e, v in r["fd"].items():
        rel = abs(r["ad"] - v) / max(abs(v), 1e-300)
        print(f"FD eps={e:.3e}: {v:.10e}   |AD-FD|/|FD| = {rel:.3e}")
    print(f"FD wall time {r['t_fd_s']:.1f} s (1 compile + 4 forward runs)")


if __name__ == "__main__":
    main()
