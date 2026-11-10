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
                         compile_only=False):
    """Return a dict with alpha, L, AD dL/dalpha, FD at eps and eps/10, and
    wall times. ``run_dino_argv`` is parsed by run_dino's own parser. Runs
    under the fp64 precision policy (restored afterwards): a float32 state
    cannot resolve the FD difference. The NEMO cards force fp64 anyway.
    ``chunk = K > 0`` runs the window as nested checkpointed scans: backprop
    keeps one state per K-step chunk, then K states and one step's
    intermediates while it recomputes a chunk. Both scans start from the
    model's seeded scan carry (``seed_carry`` gives the plain step loop the
    same start, for comparison). ``compile_only`` compiles the jitted
    value-and-gradient and returns XLA's memory estimate without running it."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prior = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        return _check(run_dino_argv, n_steps, eps_rel, seed, chunk,
                      seed_carry or chunk > 0, compile_only)
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


def _check(run_dino_argv, n_steps, eps_rel, seed, chunk=0,
           seed_carry=False, compile_only=False):
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
    if gm is None or gm.eke is None or gm.eke.closure != "geometric":
        raise SystemExit("needs the GEOMETRIC closure (--gm-kappa-scheme "
                         "geometric, or a card that selects it)")
    alpha0 = float(gm.eke.geometric.alpha)
    if seed_carry:
        run.state = _seeded_state(run)

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
        if chunk:
            from legoesm.timestepping.scan_loop import integrate_scan_generic

            def one_step(c):
                return rd.advance(run, c[0], c[1], model=m), c[1] + 1

            def one_chunk(c):
                return integrate_scan_generic(
                    c, one_step, chunk, checkpoint_interval=1,
                    return_trajectory=False)[0]

            (state, _), _ = integrate_scan_generic(
                (state, jnp.asarray(0)), one_chunk, n_steps // chunk,
                checkpoint_interval=1, return_trajectory=False)
        else:
            for k in range(n_steps):
                state = rd.advance(run, state, k, model=m)
        return jnp.sum(w_T * state.T.data) + jnp.sum(w_S * state.S.data)

    if compile_only:
        t0 = time.time()
        m = jax.jit(jax.value_and_grad(loss)).lower(
            jnp.asarray(alpha0)).compile().memory_analysis()
        return {"alpha": alpha0, "steps": n_steps, "chunk": chunk,
                "dt": run.dt, "grid": run.grid_desc,
                "t_compile_s": time.time() - t0,
                "memory": {k: getattr(m, k) for k in dir(m)
                           if k.endswith("_in_bytes")
                           and not k.startswith("host_")}}

    t0 = time.time()
    # Jitted so XLA plans the whole backward. Jitting does not shrink a full
    # 1/4 deg step's backward (~36 GiB from step 2 on); 1-step figures are low
    # because step 1 starts from a constant state (jobs 66693365, 66708115).
    L, dL = jax.jit(jax.value_and_grad(loss))(jnp.asarray(alpha0))
    dL = float(jax.block_until_ready(dL))
    t_grad = time.time() - t0
    # Peak after the gradient, before the (smaller) FD forward runs; None on CPU.
    peak = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")

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
                             own.chunk,
                             compile_only=own.compile_only)
    print(f"{r['grid']}, dt={r['dt']} s, {r['steps']} steps, "
          f"alpha={r['alpha']}, "
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
    print(f"AD dL/dalpha = {r['ad']:.10e}   ({r['t_grad_s']:.1f} s)")
    for e, v in r["fd"].items():
        rel = abs(r["ad"] - v) / max(abs(v), 1e-300)
        print(f"FD eps={e:.3e}: {v:.10e}   |AD-FD|/|FD| = {rel:.3e}")
    print(f"FD wall time {r['t_fd_s']:.1f} s (1 compile + 4 forward runs)")


if __name__ == "__main__":
    main()
