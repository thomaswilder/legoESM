"""ML-GEOMETRIC through the DINO driver (scripts/run/run_dino.py): the
committed 1 degree deck, the U-Net hook, and a short forward run on CPU.

Namelist values here are TEST-ONLY; the deck leaves them null.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_dino", _REPO / "scripts" / "run" / "run_dino.py")
rd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rd)

DECK = (_REPO / "scripts" / "experiment" / "dino"
        / "nemo_faithful_kamm_mlf_ml_geometric.yaml")
# TEST-ONLY values.
TEST_NAMELIST = ["--ml-geometric-alpha", "0.04", "--ml-geometric-sf-min", "0.1",
                 "--ml-geometric-sf-max", "1.0",
                 "--ml-geometric-ross-min-m", "2000",
                 "--ml-geometric-kappa-min", "10",
                 "--ml-geometric-kappa-max", "3000"]


def _build(monkeypatch, argv):
    from legoesm.core.precision import get_policy, set_policy
    prev = get_policy()
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", str(DECK), *argv])
    try:
        return rd.build_run(rd._parse_args())
    finally:
        set_policy(prev)


def test_deck_selects_ml_geometric_and_is_rejected_until_namelist_given(
        monkeypatch):
    with pytest.raises(ValueError) as e:
        _build(monkeypatch, [])
    for name in ("rn_geom", "rn_SFmin", "rn_SFmax", "rn_ross_min",
                 "rn_aeiv_min", "rn_aeiv_max"):
        assert name in str(e.value)


@pytest.mark.slow
def test_two_step_forward_run_with_ml_geometric(monkeypatch, tmp_path):
    """Two 1 degree steps on CPU: U-Net EKE written before step 1 and carried
    through the step unchanged, state finite, 3-D kappa within its bounds
    (0 on dry cells; <= kappa_max * max(1, sf_max) on wet ones)."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_density_and_jacobian,
        native_ml_geometric_kappa_for_state,
    )

    from tests.unit.test_eke_unet import _tiny_npz

    if not jax.config.x64_enabled:
        pytest.skip("needs JAX_ENABLE_X64=1")
    run = _build(monkeypatch, [
        *TEST_NAMELIST,
        "--ml-geometric-weights", str(_tiny_npz(tmp_path / "tiny.npz")),
        "--ml-geometric-wall-halo", "0"])
    assert run.steps_per_day == 32                    # 86400 / 2700
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        state = run.state
        eke0 = state.eke.data
        wet = np.asarray(state.land_mask.data) > 0.5
        assert state.eke.units == "m^2/s^2"
        assert np.all(np.asarray(eke0)[wet] > 0) and np.all(np.asarray(eke0)[~wet] == 0)
        for k in range(2):
            state = rd.advance(run, state, k)
        assert jnp.array_equal(state.eke.data, eke0)      # read, never written
        for f in (state.T, state.S, state.u, state.v, state.eta):
            assert bool(jnp.all(jnp.isfinite(f.data)))

        c = run.model.config
        rho, jac = gm_redi_density_and_jacobian(
            state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
            run.grid, run.z, eos=c.eos, eos_linear=c.eos_linear,
            eos_nemo_seos=c.eos_nemo_seos, mask=state.land_mask.data,
            rho_0=c.constants.rho_0, g=c.constants.g, eos_depth=c.eos_depth)
        kappa = np.asarray(native_ml_geometric_kappa_for_state(
            rho, state.T.data, state.S.data, state.land_mask.data,
            state.u_mask.data, state.v_mask.data, run.z, run.grid, c.gm_redi,
            make_eos_fn(c.eos, c.eos_linear, eos_nemo_seos=c.eos_nemo_seos,
                        rho0=c.constants.rho_0),
            jac, state.eta.data, state.H_bathy.data, None, None, None, None,
            jnp.broadcast_to(run.grid.f, state.land_mask.data.shape),
            c.constants.rho_0, c.constants.g, c.omega, state.eke.data))
    finally:
        set_policy(prev)
    act = np.asarray(run.z.is_active) & wet[..., None]
    assert np.all(np.isfinite(kappa))
    assert np.all(kappa[~act] == 0.0)
    assert kappa[act].min() > 0.0 and kappa[act].max() <= 3000.0
