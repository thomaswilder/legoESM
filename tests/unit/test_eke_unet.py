"""Unit tests for the JAX port of the Keras EKE U-Net (legoesm.ml.eke_unet).

The preprocessing and wall-fill checks are the ones ``ml_models.py`` runs on
itself, ported. A tiny random network written in the converter's ``.npz``
schema covers loading, shapes, jit and gradients without TensorFlow. Parity
with TensorFlow always runs on the committed tiny TF reference
(``fixtures/eke_unet_tiny_tf_reference.npz``, written by the converter's
``--tiny-reference``), and on the real model when its converted weights are
present.
"""

import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ml.eke_unet import (
    load_eke_unet,
    predict_eke,
    preprocess,
    surface_vorticity_over_f,
    wall_fill_index,
)

jax.config.update("jax_enable_x64", True)

WEIGHTS = os.environ.get("LEGOESM_EKE_UNET_NPZ") or str(
    pathlib.Path(__file__).resolve().parents[2]
    / "results" / "eke_unet" / "model_20260929-121036.npz")
TINY_TF = pathlib.Path(__file__).parent / "fixtures" / "eke_unet_tiny_tf_reference.npz"


def _tiny_npz(path, base=4, depth=2, seed=0):
    """Random weights in the converter's key schema."""
    rng = np.random.default_rng(seed)
    out = {"bn_epsilon": np.float32(1e-3), "features": np.array(["vor"]),
           "target": np.array(["fine_ke"]), "vor_norm": np.array([-0.0025, 0.1423])}

    def block(prefix, cin, cout):
        for j, ci in enumerate((cin, cout)):
            out[f"{prefix}_{j}_kernel"] = rng.normal(0, 0.3, (3, 3, ci, cout)).astype(np.float32)
            out[f"{prefix}_{j}_gamma"] = rng.uniform(0.5, 1.5, cout).astype(np.float32)
            out[f"{prefix}_{j}_beta"] = rng.normal(0, 0.1, cout).astype(np.float32)
            out[f"{prefix}_{j}_mean"] = rng.normal(0, 0.1, cout).astype(np.float32)
            out[f"{prefix}_{j}_var"] = rng.uniform(0.5, 1.5, cout).astype(np.float32)

    cin = 1
    for i in range(depth):
        block(f"enc{i}", cin, base * 2**i)
        cin = base * 2**i
    block("bottleneck", cin, base * 2**depth)
    cin = base * 2**depth
    for n, i in enumerate(reversed(range(depth))):
        block(f"dec{n}", cin + base * 2**i, base * 2**i)
        cin = base * 2**i
    out["out_kernel"] = rng.normal(0, 0.3, (1, 1, cin, 1)).astype(np.float32)
    np.savez(path, **out)
    return path


def _fields(ny, nx):
    jj, ii = np.meshgrid(np.arange(ny) / ny, np.arange(nx) / nx, indexing="ij")
    vor = 0.1 * np.sin(6 * np.pi * ii) * np.cos(4 * np.pi * jj)
    mask = np.ones((ny, nx))
    mask[:3] = 0.0
    return vor, mask


def test_preprocess_normalises_and_zeroes_land():
    vor, mask = _fields(20, 9)
    feats = np.asarray(preprocess(vor, mask))
    ocean = mask > 0
    assert feats.shape == (20, 9, 1)
    np.testing.assert_allclose(feats[..., 0][ocean], ((vor + 0.0025) / 0.1423)[ocean], rtol=1e-12)
    assert np.all(feats[..., 0][~ocean] == 0.0)


def test_preprocess_clips_out_of_range_inputs():
    _, mask = _fields(8, 8)
    vor = np.where(np.arange(8)[None, :] < 4, 8.0, -8.0) * np.ones_like(mask)
    feats = np.asarray(preprocess(vor, mask))
    assert feats.max() == 5.0 and feats.min() == -5.0


def test_wall_fill_reflects_about_wall_and_keeps_open_sides():
    # ml_models' own fill check: closed halo and wall at low j, other sides open
    halo = 22
    mask = np.ones((halo + 8, 3))
    r1 = halo + 1
    mask[:r1] = 0.0
    a = np.arange(float(mask.shape[0]))[:, None] * mask
    f = a[wall_fill_index(mask, halo)]
    assert f[r1 - 3:r1 + 2, 0].tolist() == [r1 + 1, r1, r1, r1, r1 + 1]
    assert np.array_equal(f[r1:], a[r1:])


def test_tiny_network_shapes_jit_and_gradients(tmp_path):
    net = load_eke_unet(_tiny_npz(tmp_path / "tiny.npz"))
    vor, mask = (jnp.asarray(a, jnp.float32) for a in _fields(13, 10))
    idx = wall_fill_index(np.asarray(mask), 0)

    eke = predict_eke(net, vor, mask, idx)
    assert eke.shape == (13, 10) and eke.dtype == jnp.float32
    assert bool(jnp.all(eke[:3] == 0.0)) and bool(jnp.all(eke[3:] > 0.0))
    # jit vs eager in float64, where XLA's float32 reordering (~1e-5 here) cannot hide a tracer bug
    net64 = jax.tree.map(lambda a: a.astype(jnp.float64), net)
    a64 = [x.astype(jnp.float64) for x in (vor, mask)]
    np.testing.assert_allclose(jax.jit(predict_eke)(net64, *a64, idx),
                               predict_eke(net64, *a64, idx), rtol=1e-12, atol=0)

    # float64 fields (x64 model state) run through the float32 network, result back in float64
    assert predict_eke(net, *a64, idx).dtype == jnp.float64

    grads = jax.grad(lambda n: jnp.mean(predict_eke(n, vor, mask, idx)))(net)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads))
    # BatchNorm moving statistics are frozen; everything else gets a gradient
    blocks = [b for pair in (*net.enc, net.bottleneck, *net.dec) for b in pair]
    gblocks = [b for pair in (*grads.enc, grads.bottleneck, *grads.dec) for b in pair]
    assert len(blocks) == 10
    for g in gblocks:
        assert not jnp.any(g.mean) and not jnp.any(g.var)
        assert all(bool(jnp.any(a != 0)) for a in (g.kernel, g.gamma, g.beta))
    assert bool(jnp.any(grads.out_kernel != 0))


def test_load_refuses_other_models(tmp_path):
    path = _tiny_npz(tmp_path / "ok.npz")
    w = dict(np.load(path))
    for key, bad in (("features", np.array(["coarse_ke_log", "vor", "sa"])),
                     ("vor_norm", np.array([0.0, 1.0]))):
        np.savez(tmp_path / "bad.npz", **{**w, key: bad})
        with pytest.raises(ValueError, match="this port takes"):
            load_eke_unet(tmp_path / "bad.npz")
    np.savez(tmp_path / "old.npz", **{k: v for k, v in w.items() if k != "target"})
    with pytest.raises(ValueError, match="no 'target' record"):
        load_eke_unet(tmp_path / "old.npz")


def test_periodic_x_makes_prediction_shift_equivariant(tmp_path):
    # all ocean, 40 columns (+2*24 wrap = 88, a multiple of 4): rolling the input by a
    # multiple of the pool stride rolls the output; with closed edges it does not
    net = load_eke_unet(_tiny_npz(tmp_path / "tiny.npz"))
    rng = np.random.default_rng(3)
    vor = jnp.asarray(0.3 * rng.standard_normal((16, 40)), jnp.float32)
    mask = jnp.ones((16, 40), jnp.float32)
    idx = wall_fill_index(np.asarray(mask), 0)
    for periodic, close in ((True, True), (False, False)):
        a = predict_eke(net, vor, mask, idx, periodic_x=periodic)
        b = predict_eke(net, jnp.roll(vor, 8, axis=1), mask, idx, periodic_x=periodic)
        assert np.allclose(np.roll(a, 8, axis=1), b, rtol=1e-5, atol=0) == close


def test_parity_with_committed_tiny_tensorflow_reference():
    w = np.load(TINY_TF)
    net = load_eke_unet(TINY_TF)
    # network alone, native shape
    got = np.asarray(net(jnp.asarray(w["ref_input"])))
    ref = w["ref_output"]
    assert np.abs(got - ref).max() <= 1e-5 * np.abs(ref).max()
    # online path: odd shape, wall in the halo, an island, one input past the clip;
    # TF side ran ml_models' own fill_reflect_about_wall and pad_to_multiple
    mask = w["online_mask"]
    eke = np.asarray(predict_eke(net, jnp.asarray(w["online_vor"], jnp.float32), jnp.asarray(mask, jnp.float32),
                                 wall_fill_index(mask, int(w["online_halo"]))))
    ref = w["online_eke"]
    assert np.all(eke[mask == 0] == 0.0)
    # compared in log EKE: the random tiny net reaches log EKE ~19, where float32
    # round-off is ~1e-5 absolute (measured max 1.2e-5)
    np.testing.assert_allclose(np.log(eke[mask > 0]), np.log(ref[mask > 0]), rtol=0, atol=5e-5)


def test_surface_vorticity_over_f_matches_definition():
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre

    grid = create_latlon_grid(n_lat=16, n_lon=32, radius=constants.R_earth)
    rng = np.random.default_rng(0)
    u = jnp.asarray(0.1 * rng.standard_normal((16, 33, 3)))
    v = jnp.asarray(0.1 * rng.standard_normal((17, 32, 3)))
    got = surface_vorticity_over_f(u, v, grid)
    zeta = relative_vorticity_cell_centre(u[..., 0], v[..., 0], grid)
    f = np.asarray(grid.grid_coriolis)
    assert got.shape == (16, 32) and np.all(f != 0)
    np.testing.assert_allclose(got, zeta / f, rtol=1e-12)
    g = jax.grad(lambda uu: jnp.sum(surface_vorticity_over_f(uu, v, grid)))(u)
    assert bool(jnp.isfinite(g).all()) and bool(jnp.any(g[..., 0] != 0)) and not jnp.any(g[..., 1:])


def test_surface_vorticity_over_f_floors_f_at_the_equator():
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre

    grid = create_latlon_grid(n_lat=15, n_lon=8, radius=constants.R_earth)  # middle row on the equator
    f = np.asarray(grid.grid_coriolis)
    small = np.abs(f) < 1e-7
    assert small[7].all() and small.sum() == 8
    rng = np.random.default_rng(1)
    u = jnp.asarray(0.1 * rng.standard_normal((15, 9)))
    v = jnp.asarray(0.1 * rng.standard_normal((16, 8)))
    got = np.asarray(surface_vorticity_over_f(u, v, grid))
    zeta = np.asarray(relative_vorticity_cell_centre(u, v, grid))
    np.testing.assert_allclose(got[small], zeta[small] / np.where(f[small] < 0, -1e-7, 1e-7), rtol=1e-12)
    np.testing.assert_allclose(got[~small], zeta[~small] / f[~small], rtol=1e-12)
    g = jax.grad(lambda uu: jnp.sum(surface_vorticity_over_f(uu, v, grid)))(u)
    assert bool(jnp.isfinite(g).all())


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason=f"converted weights not at {WEIGHTS}")
def test_parity_with_tensorflow_reference():
    w = np.load(WEIGHTS)
    got = np.asarray(load_eke_unet(WEIGHTS)(jnp.asarray(w["ref_input"])))
    ref = w["ref_output"]
    assert got.shape == ref.shape and got.dtype == np.float32
    # float32 against TF float32; measured 1.9e-6 of max|ref| on 2026-11-11 (model_20260929-121036)
    assert np.abs(got - ref).max() <= 1e-5 * np.abs(ref).max()
