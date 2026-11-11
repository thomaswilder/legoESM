"""Direct test of scripts/validate/dino_alpha_gradient_check.py."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "dino_alpha_gradient_check",
    _REPO / "scripts" / "validate" / "dino_alpha_gradient_check.py")
gc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gc)


def test_alpha_gradient_matches_finite_difference_and_restores_policy():
    """Small CPU DINO (TKE: KPP's gradient is NaN, see the script notes) with
    a raised cold-start EKE so kappa_gm sits off its lower bound and alpha
    moves the state; AD must agree with central FD and be non-zero."""
    import jax
    from legoesm.core.precision import get_policy

    if not jax.config.x64_enabled:
        pytest.skip("needs JAX_ENABLE_X64=1")
    prior = get_policy()
    r = gc.alpha_gradient_check(
        ["--gm-kappa-scheme", "geometric", "--vmix", "tke",
         "--geometric-e0-per-depth", "1e-3", "--n-lon", "8"],
        n_steps=2, eps_rel=1.0e-4)
    assert get_policy() == prior
    fd = r["fd"][max(r["fd"])]
    assert math.isfinite(r["ad"]) and r["ad"] != 0.0
    # 1e-3: FD round-off alone spreads the two FD values by ~3e-4 here.
    assert abs(r["ad"] - fd) / abs(fd) < 1.0e-3


@pytest.mark.slow
def test_alpha_gradient_finite_over_two_steps_on_nemo_faithful_card():
    """Two steps of the 1 degree NEMO-faithful card (TKE, NEMO-literal
    branches). TKE is exactly 0 at dry W points; a plain sqrt there made the
    step-2 gradient NaN (GPU job 66418927) while FD stayed clean."""
    import jax

    if not jax.config.x64_enabled:
        pytest.skip("needs JAX_ENABLE_X64=1")
    card = _REPO / "scripts" / "experiment" / "dino" / "nemo_faithful_kamm_mlf.yaml"
    r = gc.alpha_gradient_check(
        ["--config", str(card), "--gm-kappa-scheme", "geometric",
         "--geometric-e0-per-depth", "1e-3"],
        n_steps=2, eps_rel=1.0e-3)
    fd = r["fd"][max(r["fd"])]
    assert math.isfinite(r["ad"]) and r["ad"] != 0.0
    assert abs(r["ad"] - fd) / abs(fd) < 1.0e-3


def test_chunk_must_divide_steps():
    with pytest.raises(SystemExit, match="divide"):
        gc.alpha_gradient_check(["--n-lon", "8"], n_steps=4, chunk=3)


@pytest.mark.slow
def test_chunked_checkpointed_gradient_matches_plain_loop():
    """Same small DINO, 4 steps from the seeded carry: nested checkpointed
    scans (chunks of 2) must give the plain loop's loss and dL/dalpha."""
    import jax

    if not jax.config.x64_enabled:
        pytest.skip("needs JAX_ENABLE_X64=1")
    argv = ["--gm-kappa-scheme", "geometric", "--vmix", "tke",
            "--geometric-e0-per-depth", "1e-3", "--n-lon", "8"]
    ref = gc.alpha_gradient_check(argv, n_steps=4, seed_carry=True)
    r = gc.alpha_gradient_check(argv, n_steps=4, chunk=2)
    assert ref["ad"] != 0.0
    assert r["L"] == ref["L"]
    assert abs(r["ad"] - ref["ad"]) / abs(ref["ad"]) < 1.0e-10


@pytest.mark.slow
def test_unet_weight_gradient_matches_finite_difference_on_ml_geometric_card(
        tmp_path):
    """ML-GEOMETRIC on the 1 degree NEMO-faithful card: the U-Net runs once at
    the window start (outside the step), its surface EKE sets the 3-D GM
    coefficient, and dL/d(one U-Net weight, network cast to float64) after
    two steps must match central FD and be non-zero. Tiny random network and
    TEST-ONLY namelist values (kappa cap high so it rarely binds). The weight
    is on output channel 1: from rest the input is uniform and channel 0 of
    the tiny net's last features is identically 0 (ReLU), so its weight has
    no effect at all (FD exactly 0)."""
    import jax

    from tests.unit.test_eke_unet import _tiny_npz

    if not jax.config.x64_enabled:
        pytest.skip("needs JAX_ENABLE_X64=1")
    card = (_REPO / "scripts" / "experiment" / "dino"
            / "nemo_faithful_kamm_mlf_ml_geometric.yaml")
    r = gc.alpha_gradient_check(
        ["--config", str(card),
         "--ml-geometric-alpha", "0.04", "--ml-geometric-sf-min", "0.1",
         "--ml-geometric-sf-max", "1.0", "--ml-geometric-ross-min-m", "2000",
         "--ml-geometric-kappa-min", "10", "--ml-geometric-kappa-max", "1e5",
         "--ml-geometric-weights", str(_tiny_npz(tmp_path / "tiny.npz")),
         "--ml-geometric-wall-halo", "0"],
        n_steps=2, eps_rel=1.0e-3, unet_index=(0, 0, 1, 0))
    # The smaller FD step: at eps_rel 1e-3 a cap/switch kink sits inside the
    # stencil (FD off by 1.8%); at 1e-4 AD and FD agree to 4.4e-6.
    fd = r["fd"][min(r["fd"])]
    assert math.isfinite(r["ad"]) and r["ad"] != 0.0
    assert abs(r["ad"] - fd) / abs(fd) < 1.0e-3
