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


def test_more_devices_than_jax_sees_is_an_error():
    """Asking for more devices than exist must stop, not quietly run on fewer."""
    import jax

    with pytest.raises(SystemExit, match="JAX sees"):
        gc.alpha_gradient_check(["--n-lon", "8"], n_steps=1,
                                n_devices=len(jax.devices()) + 1)


@pytest.mark.slow
def test_longitude_split_over_two_cpu_devices_matches_one_device():
    """Same small DINO as above on 2 fake CPU devices: the state must really
    be split along longitude and dL/dalpha must equal the 1-device value."""
    import os
    import subprocess
    import sys

    code = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gc", {str(_SPEC.origin)!r})
gc = importlib.util.module_from_spec(spec); spec.loader.exec_module(gc)
argv = ["--gm-kappa-scheme", "geometric", "--vmix", "tke",
        "--geometric-e0-per-depth", "1e-3", "--n-lon", "8"]
rd = gc._load_run_dino(); sys.argv = ["run_dino", *argv]
s = gc._split_along_longitude(rd.build_run(rd._parse_args()).state, 2)
assert [x.data.shape[1] for x in s.T.data.addressable_shards] == [4, 4]
a1 = gc.alpha_gradient_check(argv, n_steps=2, n_devices=1)["ad"]
a2 = gc.alpha_gradient_check(argv, n_steps=2, n_devices=2)["ad"]
assert a1 != 0.0 and abs(a1 - a2) / abs(a1) < 1e-10, (a1, a2)
print("OK", a1, a2)
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=2")
    p = subprocess.run([sys.executable, "-c", code], env=env, cwd=_REPO,
                       capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0 and "OK" in p.stdout, p.stderr[-3000:]
