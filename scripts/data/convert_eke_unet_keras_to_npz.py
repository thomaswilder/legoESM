"""Convert the Keras EKE U-Net (GEOMETRIC closure, NEMO DINO) to a plain .npz.

Runs in the TensorFlow environment that trained the model, NOT the legoESM
venv. Writes every weight under the names :func:`legoesm.ml.eke_unet.load_eke_unet`
reads, the input/target names and normalisation from the model's provenance
(checked at load), and a TensorFlow reference pair (``ref_input``,
``ref_output``) on a seeded random input of the model's native shape, so the
JAX port can be checked against TensorFlow without TensorFlow installed.

Usage::

    /home/users/twilder/Python/AI4PEX/tensorflow_env/bin/python \
        scripts/data/convert_eke_unet_keras_to_npz.py \
        --keras .../hold_out_features/model_20260929-121036.keras \
        --code-dir /home/users/twilder/Python/AI4PEX/machine_learning \
        --out results/eke_unet/model_20260929-121036.npz

``--tiny-reference OUT`` instead writes the small committed test fixture: a
seeded random U-Net of the same class (4 base filters) with randomised
BatchNorm statistics, its weights, the native-shape reference pair, and one
pair through the online NEMO path (``ml_models``' own wall fill and padding)
on an odd-shaped field with land.
"""

import argparse
import ast
import json
import os
import sys
import zipfile

import numpy as np

# the only model the JAX port describes: one vorticity/f input, log fine-KE target
FEATURES, TARGET = ["vor"], ["fine_ke"]


def export_weights(model, keras):
    """Every weight of ``model`` under the npz key schema; asserts the architecture."""
    # the JAX port fixes the architecture; refuse a model it does not describe
    assert not model.use_attention, "attention blocks are not ported"
    assert tuple(model._input_shape)[-1] == 1, f"one input channel expected: {model._input_shape}"
    out = {}
    eps = set()

    def block(prefix, seq):
        layers = seq.layers
        pads = [x for x in layers if type(x).__name__ == "ReplicationPadding2D"]
        convs = [x for x in layers if isinstance(x, keras.layers.Conv2D)]
        bns = [x for x in layers if isinstance(x, keras.layers.BatchNormalization)]
        relus = [x for x in layers if isinstance(x, keras.layers.ReLU)]
        assert len(pads) == len(convs) == len(bns) == len(relus) == 2, prefix
        assert [type(x).__name__ for x in layers[:8]] == [
            "ReplicationPadding2D", "Conv2D", "BatchNormalization", "ReLU"] * 2, prefix
        for relu in relus:
            c = relu.get_config()
            assert c["max_value"] is None and not c["negative_slope"] and not c["threshold"], prefix
        for j, (pad, conv, bn) in enumerate(zip(pads, convs, bns)):
            assert tuple(pad.padding) == (1, 1), prefix
            assert conv.kernel_size == (3, 3) and conv.padding == "valid", prefix
            assert conv.dilation_rate == (1, 1) and conv.activation.__name__ == "linear", prefix
            assert not conv.use_bias and conv.strides == (1, 1), prefix
            assert bn.center and bn.scale and bn.axis in (-1, 3), prefix
            eps.add(bn.epsilon)
            out[f"{prefix}_{j}_kernel"] = conv.kernel.numpy()  # (kh, kw, in, out)
            out[f"{prefix}_{j}_gamma"] = bn.gamma.numpy()
            out[f"{prefix}_{j}_beta"] = bn.beta.numpy()
            out[f"{prefix}_{j}_mean"] = bn.moving_mean.numpy()
            out[f"{prefix}_{j}_var"] = bn.moving_variance.numpy()

    for i, seq in enumerate(model.encoder_blocks):
        block(f"enc{i}", seq)
    block("bottleneck", model.bottleneck)
    for i, seq in enumerate(model.decoder_blocks):
        block(f"dec{i}", seq)
    for pool in model.pool_layers:
        assert pool.pool_size == (2, 2) and pool.strides == (2, 2) and pool.padding == "valid"
    for up in model.upsample_layers:
        assert up.size == (2, 2) and up.interpolation == "nearest"
    oc = model.output_conv
    assert oc.kernel_size == (1, 1) and not oc.use_bias and oc.filters == 1
    assert oc.activation.__name__ == "linear"
    out["out_kernel"] = oc.kernel.numpy()

    assert len(eps) == 1, f"BatchNorm epsilons differ: {eps}"
    out["bn_epsilon"] = np.float32(eps.pop())

    # TensorFlow reference on the model's native input shape
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, *model._input_shape)).astype(np.float32)
    out["ref_input"] = x[0]
    out["ref_output"] = np.asarray(model(x, training=False))[0]
    return out


def model_meta(features, target, vor_stats):
    """Input/target names and vorticity/f (mean, std), as ``load_eke_unet`` checks them."""
    assert list(features) == FEATURES and list(target) == TARGET, (features, target)
    return {"features": np.array(features), "target": np.array(target),
            "vor_norm": np.array([vor_stats["mean"], vor_stats["std"]], np.float64)}


def read_provenance(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("assets/provenance.json"))


def _online_ml_models(code_dir):
    """``pad_to_multiple`` and ``fill_reflect_about_wall`` from the online NEMO
    ``ml_models.py``, executed from source without its module-level model load."""
    from scipy.ndimage import distance_transform_edt
    src = open(os.path.join(code_dir, "inference_testing", "DINO", "ml_models.py")).read()
    keep = [n for n in ast.parse(src).body
            if (isinstance(n, ast.FunctionDef) and n.name in ("pad_to_multiple", "fill_reflect_about_wall"))
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("CLIP_SIGMA", "HALO"))]
    ns = {"np": np, "distance_transform_edt": distance_transform_edt}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "ml_models", "exec"), ns)
    return ns


def tiny_reference(code_dir, keras, model_components, unet):
    """Seeded tiny U-Net, its weights and two TF reference pairs (see module doc)."""
    keras.utils.set_random_seed(0)
    sc = model_components.Scenario(input_var=FEATURES, target=TARGET, base_filters=4, depth=2)
    model = unet.UNet(sc, input_shape=(40, 40, 1), dropout_rate=0.0)
    model(np.zeros((1, 40, 40, 1), np.float32), training=False)
    rng = np.random.default_rng(0)
    for layer in (x for seq in [*model.encoder_blocks, model.bottleneck, *model.decoder_blocks]
                  for x in seq.layers):
        if isinstance(layer, keras.layers.BatchNormalization):  # default 0/1 stats test nothing
            n = layer.moving_mean.shape
            layer.moving_mean.assign(rng.normal(0, 0.2, n).astype(np.float32))
            layer.moving_variance.assign(rng.uniform(0.3, 2.0, n).astype(np.float32))
            layer.gamma.assign(rng.uniform(0.5, 1.5, n).astype(np.float32))
            layer.beta.assign(rng.normal(0, 0.2, n).astype(np.float32))
    stats = {"mean": -0.0025, "std": 0.1423}
    out = export_weights(model, keras) | model_meta(FEATURES, TARGET, stats)

    # one field through the online path: odd shape, closed south wall in the halo, an island
    ml = _online_ml_models(code_dir)
    halo = 5
    ml["HALO"] = halo
    ny, nx = 37, 29
    jj, ii = np.meshgrid(np.arange(ny) / ny, np.arange(nx) / nx, indexing="ij")
    vor = 0.4 * np.sin(6 * np.pi * ii) * np.cos(4 * np.pi * jj) + 0.2 * rng.standard_normal((ny, nx))
    vor[20, 10] = 3.0  # beyond the clip
    mask = np.ones((ny, nx))
    mask[:halo + 1] = 0.0
    mask[15:19, 8:13] = 0.0
    a = np.clip((vor - stats["mean"]) / stats["std"] * mask, -ml["CLIP_SIGMA"], ml["CLIP_SIGMA"])
    a = ml["fill_reflect_about_wall"]([a], mask)[0]
    a_p, _ = ml["pad_to_multiple"](a, 4)
    log_eke = np.asarray(model(a_p[None, ..., None].astype(np.float32), training=False))[0, :ny, :nx, 0]
    out |= {"online_vor": vor, "online_mask": mask, "online_halo": np.int64(halo),
            "online_eke": np.where(mask > 0, np.exp(log_eke), 0.0)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--keras", help=".keras model file")
    p.add_argument("--code-dir", required=True,
                   help="training repo holding unet.py and model_components.py "
                        "(they register the custom Keras classes)")
    p.add_argument("--out", help="output .npz")
    p.add_argument("--tiny-reference", metavar="OUT",
                   help="write the small committed test fixture instead")
    args = p.parse_args()
    if bool(args.tiny_reference) == bool(args.keras and args.out):
        p.error("give either --keras and --out, or --tiny-reference")

    sys.path.insert(0, args.code_dir)
    import keras
    import model_components  # registers cnn>Scenario, ReplicationPadding2D
    import unet  # registers unet>UNet

    if args.tiny_reference:
        out, path = tiny_reference(args.code_dir, keras, model_components, unet), args.tiny_reference
    else:
        prov = read_provenance(args.keras)["run_settings"]
        assert not prov.get("n_lags"), f"lagged inputs are not ported: n_lags={prov['n_lags']}"
        model = keras.saving.load_model(args.keras, compile=False)
        out = export_weights(model, keras) | model_meta(
            prov["features"], prov["target"], prov["norm_stats"]["vor"])
        path = args.out

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **out)
    n = sum(v.size for k, v in out.items()
            if k.endswith(("_kernel", "_gamma", "_beta", "_mean", "_var")))
    print(f"wrote {path}: {len(out)} arrays, {n} weights, bn_epsilon {out['bn_epsilon']}")


if __name__ == "__main__":
    main()
