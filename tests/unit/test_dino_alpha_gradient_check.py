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
