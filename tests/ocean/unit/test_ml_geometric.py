"""ML-GEOMETRIC column formulas (eke.ml_geometric_*) against a hand-computed
column, including the quirks of the user's NEMO ldfeke.F90 that the port
keeps as written, plus the config / dispatch guards.

Parameter values here are TEST-ONLY (the real ones come from the run's
&namldf_eke namelist, not yet supplied).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.eke import (
    EKEConfig,
    GeometricConfig,
    MLGeometricConfig,
    ml_geometric_column_integrals,
    ml_geometric_depth_integrated_eke,
    ml_geometric_kappa_surface,
    ml_geometric_kappa_t,
    ml_geometric_kappa_w,
    ml_geometric_structure_function_t,
    ml_geometric_structure_function_w,
    validate_eke_config,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    validate_ml_geometric_gm_config,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)

# TEST-ONLY values (not the namelist's).
ML = MLGeometricConfig(alpha=0.05, sf_min=0.1, sf_max=1.5, ross_min_m=1000.0,
                       kappa_min=1.0, kappa_max=1000.0)
OMEGA = constants.Omega

# One column, 6 levels: cells 0..3 wet (mbkt = 4), mixed layer down to W
# level 2 (NEMO nmln = 2, 1-based). W index k = top of cell k.
N2 = np.array([5e-5, 1e-5, 2e-5, 4e-5, 0.0, 0.0])   # [0] must be ignored
NMLN, MBKT = np.array(2), np.array(4)
TMASK = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
WMASK = TMASK                                           # tmask(k)*tmask(k-1)
E3T = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
E3W = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0])
WSLPI = np.array([0.0, 1e-3, 2e-3, 1e-3, 0.0, 0.0])
WSLPJ = np.array([0.0, 0.0, 1e-3, 0.0, 0.0, 0.0])
ZCK = np.array([1.0, 1.0, 0.5, 0.05, 0.0, 0.0])         # 0.05 hits the 0.1 floor

# Hand-computed: ik = MIN(nmln, mbkt) + 1 = 3 (1-based) -> N2_ref = N2[2].
SF_W = np.array([0.0,          # surface: the Fortran loop starts at jk = 2
                 1.5,          # jk = 2 <= nmln: sf_max
                 1.0,          # 2e-5 / 2e-5
                 1.5,          # 4e-5 / 2e-5 = 2 -> capped at sf_max
                 0.1,          # 0 -> floored at sf_min (unmasked, as written)
                 0.1])
SF_T = np.array([0.75,         # top cell: 0.5*(0 + 1.5), half weight
                 1.25, 1.25,
                 0.0,          # bottom wet cell (jk = mbkt): zero
                 0.0, 0.0])


def _col(a):
    return jnp.asarray(a)[None, :]   # one column, leading axis of 1


def test_structure_function_w_as_written():
    sf = ml_geometric_structure_function_w(_col(N2), NMLN[None], MBKT[None], ML)
    np.testing.assert_allclose(np.asarray(sf)[0], SF_W, rtol=1e-14)


def test_structure_function_t_top_half_bottom_zero():
    sf_t = ml_geometric_structure_function_t(_col(SF_W), MBKT[None])
    np.testing.assert_allclose(np.asarray(sf_t)[0], SF_T, rtol=1e-14)


def test_mixed_layer_to_the_seafloor_is_sf_max_on_every_wet_level():
    # nmln = mbkt + 1 (zdfmxl's bottom cap): no level is below the ML.
    sf = ml_geometric_structure_function_w(_col(N2), np.array([5]), MBKT[None], ML)
    assert np.all(np.asarray(sf)[0, 1:5] == ML.sf_max)


def test_depth_integrated_eke_hand_computed():
    e = ml_geometric_depth_integrated_eke(jnp.asarray([0.1]), _col(SF_T),
                                          _col(E3T), _col(TMASK))
    # 0.1 * (0.75*10 + 1.25*20 + 1.25*30) = 7.0
    assert float(e[0]) == pytest.approx(7.0, rel=1e-14)


def test_column_integrals_hand_computed():
    zn, zn_slp = ml_geometric_column_integrals(
        _col(N2), _col(WSLPI), _col(WSLPJ), _col(ZCK), _col(SF_W),
        _col(E3W), _col(TMASK))
    # Levels jk >= 2 only (lego k = 1..3 wet); N2[0] = 5e-5 must not count.
    zn_ref = (math.sqrt(1e-5) * 15 + math.sqrt(2e-5) * 25
              + math.sqrt(4e-5) * 35)
    zn_slp_ref = (
        15 * 1.5 * math.sqrt(1e-5 * (1e-3 / 1.0) ** 2)
        + 25 * 1.0 * math.sqrt(2e-5 * ((2e-3 / 0.5) ** 2 + (1e-3 / 0.5) ** 2))
        + 35 * 1.5 * math.sqrt(4e-5 * (1e-3 / 0.1) ** 2))   # zck floored to 0.1
    assert float(zn[0]) == pytest.approx(zn_ref, rel=1e-13)
    assert float(zn_slp[0]) == pytest.approx(zn_slp_ref, rel=1e-13)


def _kappa_sfc(eke_int, zn_slp, zn, depth, f):
    return np.asarray(ml_geometric_kappa_surface(
        jnp.asarray(eke_int), jnp.asarray(zn_slp), jnp.asarray(zn),
        jnp.asarray(depth), jnp.asarray(f), OMEGA, jnp.ones(len(eke_int)), ML))


def test_kappa_surface_cap_tapers_floor_in_fortran_order():
    f20 = 2 * OMEGA * math.sin(math.radians(20.0))
    # Rossby radius R = 0.5*zn/|f|: zn = 1 gives R ~ 5 km >> ross_min (taper
    # 1); zn = 1e-3 gives R ~ 5 m << ross_min (taper 0, the hard switch).
    # (a) deep, extratropical: alpha*E/zn_slp times the deep coastal taper.
    # (b) raw 1e5 -> capped at kappa_max BEFORE the tropical factor 0.5.
    # (c) R << ross_min -> tapered to 0, then floored at kappa_min.
    # (d) H = 800 m: coastal taper exactly 0.5.
    k = _kappa_sfc([2.0, 2.0e3, 2.0, 2.0], [1e-3, 1e-3, 1e-3, 1e-3],
                   [1.0, 1.0, 1e-3, 1.0], [3000.0, 3000.0, 3000.0, 800.0],
                   [2 * f20, 0.5 * f20, 2 * f20, 2 * f20])
    coast = 0.5 * (1 - math.tanh((-3000.0 + 800.0) / 300.0))
    assert k[0] == pytest.approx(0.05 * 2.0 / 1e-3 * coast, rel=1e-12)
    assert k[1] == pytest.approx(ML.kappa_max * coast * 0.5, rel=1e-12)
    assert k[2] == ML.kappa_min
    assert k[3] == pytest.approx(0.05 * 2.0 / 1e-3 * 0.5, rel=1e-12)


def test_kappa_w_surface_weight_one_and_t_wet_count_average():
    kw = np.asarray(ml_geometric_kappa_w(jnp.asarray([100.0]), _col(SF_W),
                                         _col(WMASK)))[0]
    # surface weight 1 although SF_w(surface) = 0; below: SF_w * wmask
    np.testing.assert_allclose(kw, [100.0, 150.0, 100.0, 150.0, 0.0, 0.0])
    kt = np.asarray(ml_geometric_kappa_t(_col(kw), _col(WMASK), _col(TMASK)))[0]
    # (k_w(k) + k_w(k+1)) / wet W count; bottom wet cell has one wet W level.
    np.testing.assert_allclose(kt, [125.0, 125.0, 125.0, 150.0, 0.0, 0.0])


def test_gradients_finite_at_zero_stratification():
    """Guarded square roots: a level with N^2 = 0 gives finite gradients
    (a plain sqrt gives 0*inf = NaN there)."""
    n2 = jnp.asarray(N2).at[2].set(0.0)

    def total(eke, n2):
        sf = ml_geometric_structure_function_w(n2[None], NMLN[None], MBKT[None], ML)
        zn, zn_slp = ml_geometric_column_integrals(
            n2[None], _col(WSLPI), _col(WSLPJ), _col(ZCK), sf,
            _col(E3W), _col(TMASK))
        e_int = ml_geometric_depth_integrated_eke(
            eke, ml_geometric_structure_function_t(sf, MBKT[None]),
            _col(E3T), _col(TMASK))
        k = ml_geometric_kappa_surface(e_int, zn_slp, zn, jnp.asarray([3000.0]),
                                       jnp.asarray([1e-4]), OMEGA,
                                       jnp.ones(1), ML)
        return jnp.sum(ml_geometric_kappa_w(k, sf, _col(WMASK)))

    g_e, g_n2 = jax.grad(total, argnums=(0, 1))(jnp.asarray([1e-2]), n2)
    assert np.all(np.isfinite(g_e)) and np.all(np.isfinite(g_n2))
    assert float(g_e[0]) > 0.0


# --- config / dispatch guards ------------------------------------------------

def test_eke_config_accepts_ml_geometric_and_guards_it():
    validate_eke_config(EKEConfig(closure="ml_geometric", ml_geometric=ML))
    with pytest.raises(ValueError, match="requires EKEConfig.ml_geometric"):
        validate_eke_config(EKEConfig(closure="ml_geometric"))
    with pytest.raises(ValueError, match="silently ignored"):
        validate_eke_config(EKEConfig(ml_geometric=ML))
    with pytest.raises(ValueError, match="defaults"):
        validate_eke_config(EKEConfig(closure="ml_geometric", ml_geometric=ML,
                                      geometric=GeometricConfig()))
    with pytest.raises(ValueError, match="unknown EKEConfig.closure scheme"):
        validate_eke_config(EKEConfig(closure="ml_geometrik"))


def test_ml_geometric_config_has_no_defaults_and_rejects_none():
    with pytest.raises(TypeError):
        MLGeometricConfig()                                   # no defaults
    with pytest.raises(ValueError, match="rn_geom"):
        validate_eke_config(EKEConfig(closure="ml_geometric",
                                      ml_geometric=ML._replace(alpha=None)))
    with pytest.raises(ValueError, match="kappa_min <= kappa_max"):
        validate_eke_config(EKEConfig(
            closure="ml_geometric",
            ml_geometric=ML._replace(kappa_min=2.0e3)))


def test_gm_config_must_be_nemo_native():
    eke = EKEConfig(closure="ml_geometric", ml_geometric=ML)
    good = GMRediConfig(slope_scheme="nemo_iso_lap",
                        slope_positions="nemo_native",
                        gm_bolus_kappa_face_average=True, eke=eke)
    validate_ml_geometric_gm_config(good)
    for bad in (good._replace(slope_scheme="triads"),
                good._replace(slope_positions="mode_b"),
                good._replace(gm_bolus_kappa_face_average=False),
                good._replace(visbeck=good.visbeck._replace(enabled=True))):
        with pytest.raises(ValueError, match="ml_geometric"):
            validate_ml_geometric_gm_config(bad)


def test_dino_rejects_unset_namelist_values_and_unknown_scheme():
    from legoesm.ocean.experiments.dino import (
        DINOConfig,
        dino_lat_lon_grid,
        dino_lat_lon_state,
        dino_lat_lon_vertical,
        dino_ml_geometric_config,
    )
    cfg = dataclasses.replace(DINOConfig(), gm_kappa_scheme="ml_geometric",
                              ml_geometric_alpha=0.05)
    with pytest.raises(ValueError) as e:
        dino_ml_geometric_config(cfg)
    for name in ("rn_SFmin", "rn_SFmax", "rn_ross_min", "rn_aeiv_min",
                 "rn_aeiv_max"):
        assert name in str(e.value)
    assert "rn_geom" not in str(e.value)
    bogus = dataclasses.replace(DINOConfig(), gm_kappa_scheme="ml_geometrik")
    grid = dino_lat_lon_grid(bogus, n_lon=8)
    with pytest.raises(ValueError, match="unknown DINOConfig.gm_kappa_scheme"):
        dino_lat_lon_state(grid, dino_lat_lon_vertical(grid, bogus), bogus)
