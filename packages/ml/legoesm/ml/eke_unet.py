"""EKE U-Net: JAX port of the Keras U-Net that predicts surface EKE for GEOMETRIC.

The network was trained in TensorFlow/Keras (``unet.UNet`` in the AI4PEX
``machine_learning`` repo) and runs online in NEMO DINO at 1/4 deg through
eophis, where ``ml_models.ml_geometric`` turns surface fields into the
eddy kinetic energy the GEOMETRIC GM closure needs. This module reproduces
that inference path so the network can be differentiated w.r.t. its weights
inside legoESM.

The ported model is the vorticity-only one, ``model_20260929-121036``
(feature-ablation sweep, trained on SO_JET/EAST/GYRE/TROP; its provenance
lists ``features: ['vor']``). Its single input is surface relative vorticity
divided by f, as in training (``preprocess_data.normalize_data``).

Architecture (depth 2, base 64 filters):
each conv block is two of [1-cell edge-replicating pad -> 3x3 conv, no bias
-> BatchNorm (moving statistics) -> ReLU]. Encoder blocks of 64 and 128
filters each followed by a 2x2 max pool, a 256-filter bottleneck, decoder
blocks of 128 and 64 filters each preceded by 2x nearest upsampling and a
concatenation [upsampled, skip], then a 1x1 conv without bias to one channel
(log EKE). Keras' ``ReplicationPadding2D`` uses ``tf.pad(..., 'SYMMETRIC')``;
for a 1-cell pad that equals edge replication, and ``jnp.pad`` mode
``"symmetric"`` reproduces it for any width.

Only inference is ported: BatchNorm uses its moving statistics and the
spatial dropout of the encoder is off (Keras ``training=False``). The moving
statistics are frozen (no gradient reaches them), so fine-tuning trains the
convolutions and BatchNorm's scale and shift only. They are still pytree
leaves: an optimizer with weight decay must mask them out.

The vorticity/f normalisation (mean, std) belongs to the weights: each ``.npz``
carries its own ``vor_norm`` (the base model's is (-0.0025, 0.1423), computed
on 1/4 deg data; a model fine-tuned at 1 deg carries 1 deg statistics) and
:func:`load_eke_unet` puts it on the network as ``vor_mean`` / ``vor_std``.

Online, :func:`surface_vorticity_over_f` builds the input from the model's
velocities, :func:`network_input` turns it into the network's input array, and
:func:`predict_eke` runs the network on it.

Weights come from a plain ``.npz`` written by
``scripts/data/convert_eke_unet_keras_to_npz.py`` (TensorFlow is not needed
to load it).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt

# normalised inputs are clipped to +/- this many standard deviations (ml_models.CLIP_SIGMA)
CLIP_SIGMA = 5.0
# floor on |f| [1/s] in vorticity/f (user's choice, 2026-11-11)
F_MIN = 1e-7
# columns wrapped onto each side of a zonally periodic domain; >= the network's
# receptive half-width (23 cells, measured 2026-11-11), and 2 mod 4 so the pooling
# grid has the same x phase as NEMO's 22-column halo (user's choice, 2026-11-12)
WRAP_COLUMNS = 26


class ConvBN(eqx.Module):
    """Edge-replicating pad, 3x3 conv (no bias), BatchNorm (inference), ReLU on (H, W, C)."""
    kernel: jax.Array  # (3, 3, C_in, C_out), Keras HWIO layout
    gamma: jax.Array
    beta: jax.Array
    mean: jax.Array
    var: jax.Array
    eps: float = eqx.field(static=True)

    def __call__(self, x):
        x = jnp.pad(x, ((1, 1), (1, 1), (0, 0)), mode="symmetric")
        x = jax.lax.conv_general_dilated(
            x[None], self.kernel, window_strides=(1, 1), padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )[0]
        # moving statistics frozen: training them by gradient breaks BatchNorm
        # (a variance of 0.0049 vs eps 1e-3 goes NaN within a few Adam steps)
        mean, var = jax.lax.stop_gradient((self.mean, self.var))
        # same operation order as tf.nn.batch_normalization
        inv = jax.lax.rsqrt(var + self.eps) * self.gamma
        return jax.nn.relu(x * inv + (self.beta - mean * inv))


class EKEUNet(eqx.Module):
    """The EKE U-Net on one sample: (H, W, 1) normalised vorticity/f -> (H, W, 1) log EKE.

    H and W must be multiples of ``2**len(enc)``; :func:`network_input` pads to that.
    ``vor_mean``, ``vor_std``: the vorticity/f normalisation the weights were
    trained with (static, so not trained); :func:`preprocess` applies it.
    """
    enc: list   # per level, (ConvBN, ConvBN)
    bottleneck: tuple
    dec: list   # coarse to fine, (ConvBN, ConvBN)
    out_kernel: jax.Array  # (1, 1, C, 1)
    vor_mean: float = eqx.field(static=True)
    vor_std: float = eqx.field(static=True)

    def __call__(self, x):
        skips = []
        for c0, c1 in self.enc:
            x = c1(c0(x))
            skips.append(x)
            h, w, c = x.shape
            x = x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))
        c0, c1 = self.bottleneck
        x = c1(c0(x))
        for (c0, c1), skip in zip(self.dec, reversed(skips)):
            x = jnp.repeat(jnp.repeat(x, 2, axis=0), 2, axis=1)
            x = c1(c0(jnp.concatenate([x, skip], axis=-1)))
        return x @ self.out_kernel[0, 0]


def load_eke_unet(path) -> EKEUNet:
    """Build an :class:`EKEUNet` from the ``.npz`` written by the conversion script.

    Refuses a file whose model is not the one this module describes (inputs or
    target differ, the vorticity normalisation is not a finite mean with a
    finite positive std, or the file predates those records).
    """
    w = np.load(path)
    for key in ("features", "target", "vor_norm"):
        if key not in w.files:
            raise ValueError(f"{path}: no '{key}' record; reconvert with "
                             "scripts/data/convert_eke_unet_keras_to_npz.py")
    got = (list(w["features"]), list(w["target"]))
    if got != (["vor"], ["fine_ke"]):
        raise ValueError(f"{path}: model has (features, target) {got}; "
                         "this port takes (['vor'], ['fine_ke'])")
    vor_mean, vor_std = (float(a) for a in w["vor_norm"])
    if not (np.isfinite(vor_mean) and np.isfinite(vor_std) and vor_std > 0):
        raise ValueError(f"{path}: vor_norm (mean, std) = {(vor_mean, vor_std)} must be "
                         "a finite mean and a finite positive std")
    eps = float(w["bn_epsilon"])

    def block(prefix):
        return tuple(
            ConvBN(*(jnp.asarray(w[f"{prefix}_{j}_{k}"])
                     for k in ("kernel", "gamma", "beta", "mean", "var")), eps=eps)
            for j in range(2)
        )

    depth = sum(1 for k in w.files if k.startswith("enc") and k.endswith("_0_kernel"))
    return EKEUNet(
        enc=[block(f"enc{i}") for i in range(depth)],
        bottleneck=block("bottleneck"),
        dec=[block(f"dec{i}") for i in range(depth)],
        out_kernel=jnp.asarray(w["out_kernel"]),
        vor_mean=vor_mean,
        vor_std=vor_std,
    )


def preprocess(vor, mask, vor_mean, vor_std):
    """Normalised, clipped vorticity/f as (ny, nx, 1), zero on land.

    Port of ``ml_models.preprocess_inputs`` followed by the clip in
    ``ml_geometric``, on 2D (y, x) fields already in the network's
    orientation (NEMO's (i, j, 1) squeeze and transpose are left to the caller).
    """
    a = jnp.clip((vor - vor_mean) / vor_std * mask, -CLIP_SIGMA, CLIP_SIGMA)
    return a[..., None]


def wall_fill_index(mask, halo: int, *, pad_ns: bool):
    """Gather indices that reproduce ``ml_models.fill_reflect_about_wall``.

    Land takes its nearest ocean cell; a side whose outer ``halo`` cells hold
    no ocean is a closed wall, and its halo reflects about the grid edge.
    Computed once on the host from the static mask; ``x[idx]`` then applies the
    fill to a (ny, nx, ...) array inside jit. ``halo=0`` gives the plain
    nearest-ocean fill.

    ``pad_ns=False``: ``mask`` already carries its halo, as eophis hands it to
    ml_models. ``pad_ns=True``: ``mask`` has none (legoESM fields), so
    ``halo`` land rows are first added at the south and north edges, as NEMO's
    closed-side halos; the indices then address the padded (ny + 2 halo, nx)
    array, which :func:`predict_eke` builds and crops back.
    """
    mask = np.asarray(mask)
    if pad_ns:
        mask = np.pad(mask, ((halo, halo), (0, 0)))
    near = distance_transform_edt(mask == 0, return_distances=False, return_indices=True)
    ny, nx = mask.shape
    south, north = (0 if s.any() else halo for s in (mask[:halo], mask[ny - halo:]))
    west, east = (0 if s.any() else halo for s in (mask[:, :halo], mask[:, nx - halo:]))
    return tuple(np.pad(n[south:ny - north, west:nx - east],
                        ((south, north), (west, east)), mode="reflect") for n in near)


def surface_vorticity_over_f(u, v, grid):
    """The network's input from model velocities: surface relative vorticity / f
    at cell centres, (n_lat, n_lon).

    The quantity the model was trained on: the C-grid curl at cell corners,
    averaged to cell centres (``07_vorticity.py``; here legoESM's
    ``relative_vorticity_cell_centre``), divided by f at cell centres
    (``preprocess_data.normalize_data``). ``u`` (n_lat, n_lon+1[, nz]) and
    ``v`` (n_lat+1, n_lon[, nz]) are face velocities with the surface at level
    0. |f| is floored at ``F_MIN`` (sign kept, + at f == 0): the training data
    never reached the equator, and near it the input clip saturates |zeta/f|.
    The two curls agree in the interior but differ O(1) at coasts (Track A
    check V4, 2026-11-10).
    """
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre

    if u.ndim == 3:
        u, v = u[..., 0], v[..., 0]
    zeta = relative_vorticity_cell_centre(u, v, grid)
    f = jnp.asarray(grid.grid_coriolis, zeta.dtype)  # f at cell centres (grids.protocol)
    return zeta / jnp.where(f < 0, jnp.minimum(f, -F_MIN), jnp.maximum(f, F_MIN))


def network_input(net: EKEUNet, vor, mask, fill_index, periodic_x: bool = False):
    """The network's input array from surface relative vorticity / f on (ny, nx).

    Port of ``ml_models.ml_geometric`` before its network call: preprocess and
    clip with ``net``'s normalisation, wall fill (``fill_index`` from
    :func:`wall_fill_index`; built with ``pad_ns=True``, the N/S halo rows are
    added here first), ``WRAP_COLUMNS`` wrapped columns on each side when
    ``periodic_x`` (legoESM DINO's channel; else the x edges are closed), and
    an edge-pad to a multiple of ``2**depth``. Runs in the network's dtype
    (float32).

    Returns ``(x, h, w)``: ``x`` of shape (ny + 2 h + pad, nx + 2 w + pad, 1),
    ``h`` the halo rows and ``w`` the wrapped columns added on each side, so
    the field is ``x[h:h + ny, w:w + nx]`` after the network.
    """
    ny, nx = mask.shape
    # halo rows added at each N/S edge by wall_fill_index(pad_ns=True), else 0
    h, odd = divmod(fill_index[0].shape[0] - ny, 2)
    if odd or h < 0 or fill_index[0].shape[1] != nx:
        raise ValueError(f"fill_index shape {fill_index[0].shape} does not fit mask {mask.shape}")
    dt = net.out_kernel.dtype
    x = preprocess(vor.astype(dt), mask.astype(dt), net.vor_mean, net.vor_std)
    x = jnp.pad(x, ((h, h), (0, 0), (0, 0)))[fill_index]
    w = WRAP_COLUMNS if periodic_x else 0
    # shortcut: the land fill's nearest ocean cell does not look across the seam;
    # revisit if coasts next to the seam matter.
    x = jnp.pad(x, ((0, 0), (w, w), (0, 0)), mode="wrap")
    m = 2 ** len(net.enc)
    x = jnp.pad(x, ((0, -(ny + 2 * h) % m), (0, -(nx + 2 * w) % m), (0, 0)), mode="edge")
    return x, h, w


def predict_eke(net: EKEUNet, vor, mask, fill_index, periodic_x: bool = False):
    """Surface EKE on (ny, nx) from surface relative vorticity / f.

    Port of ``ml_models.ml_geometric`` minus its NEMO reshaping and its
    non-finite check (which raises, so cannot run under jit):
    :func:`network_input`, network, crop the halo, exp, zero on land. The
    result is cast back to ``vor``'s dtype.
    """
    ny, nx = mask.shape
    x, h, w = network_input(net, vor, mask, fill_index, periodic_x)
    log_eke = net(x)[h:h + ny, w:w + nx, 0]
    ocean = mask > 0
    # exp only of ocean values: an overflow on land would make the gradient NaN
    return jnp.where(ocean, jnp.exp(jnp.where(ocean, log_eke, 0.0)), 0.0).astype(vor.dtype)
