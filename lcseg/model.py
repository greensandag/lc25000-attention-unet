"""U-Net with attention gates and deep supervision (Section III.F, Equations 1-2, Table 4).

Variants (Section III.G, Table 10)
  unet          standard U-Net: no attention gates, no deep supervision
  att           Attention U-Net without deep supervision
  full          complete model: attention gates + deep supervision
  full_nofocal  complete model trained without the focal term (same architecture as `full`)
"""
from .config import SIZE

VARIANTS = ("unet", "att", "full", "full_nofocal")


def has_attention(variant):
    return variant in ("att", "full", "full_nofocal")


def has_deep_supervision(variant):
    return variant in ("full", "full_nofocal")


def build_model(variant, mcfg):
    import keras
    from keras import layers

    assert variant in VARIANTS, variant
    att, ds = has_attention(variant), has_deep_supervision(variant)

    def block(x, f, drop, name):
        # Conv2D -> BN -> ReLU -> Dropout -> Conv2D -> BN -> ReLU (dropout between the convolutions)
        x = layers.Conv2D(f, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)
        if drop:
            x = layers.Dropout(drop, name=f"{name}_drop")(x)
        x = layers.Conv2D(f, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation("relu", name=f"{name}_relu2")(x)
        return x

    def attention_gate(x, g, inter, name):
        # Eq. 1: q = psi^T(ReLU(W_x^T x + W_g^T g + b_g)) + b_psi ;  Eq. 2: alpha = sigmoid(q)
        theta = layers.Conv2D(inter, 1, use_bias=False, name=f"{name}_Wx")(x)
        phi = layers.Conv2D(inter, 1, use_bias=True, name=f"{name}_Wg")(g)       # b_g
        a = layers.Activation("relu", name=f"{name}_relu")(layers.Add(name=f"{name}_add")([theta, phi]))
        q = layers.Conv2D(1, 1, use_bias=True, name=f"{name}_psi")(a)            # b_psi
        alpha = layers.Activation("sigmoid", name=f"{name}_alpha")(q)
        return layers.Multiply(name=f"{name}_gated")([x, alpha])               # alpha_i * x_i

    inp = layers.Input((SIZE, SIZE, 3), name="image")
    skips, x = [], inp
    for lvl, (f, d) in enumerate(zip(mcfg["enc_filters"], mcfg["enc_dropout"]), start=1):
        x = block(x, f, d, f"enc{lvl}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{lvl}")(x)
    x = block(x, mcfg["bottleneck_filters"], mcfg["bottleneck_dropout"], "bottleneck")

    outputs = {}
    for k, (f, d, inter) in enumerate(zip(mcfg["dec_filters"], mcfg["dec_dropout"], mcfg["ag_inter"]), start=1):
        up = layers.Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up{k}")(x)
        s = skips[-k]
        if att:
            s = attention_gate(s, up, inter, f"ag{k}")
        x = layers.Concatenate(name=f"cat{k}")([up, s])
        x = block(x, f, d, f"dec{k}")
        if ds and k in (1, 2):
            # auxiliary outputs after the 1st (1/8 resolution) and 2nd (1/4) decoder blocks,
            # 1x1 convolution + sigmoid, resized bilinearly to 256x256
            key, scale = ("aux8", 8) if k == 1 else ("aux4", 4)
            h = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name=f"{key}_head")(x)
            outputs[key] = layers.UpSampling2D(scale, interpolation="bilinear", dtype="float32", name=key)(h)
    outputs["main"] = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="main")(x)
    order = ["main"] + [k for k in ("aux4", "aux8") if k in outputs]
    return keras.Model(inp, {k: outputs[k] for k in order}, name=variant)


def inference_model(model):
    """Graph with the main output only (used for evaluation, timing, and parameter counts)."""
    import keras
    return keras.Model(model.input, model.get_layer("main").output, name=model.name + "_infer")


def conv_flops(model):
    """Floating point operations (2 x multiply-accumulate) of all convolution layers per image."""
    from keras import layers
    total = 0
    for l in model.layers:
        if isinstance(l, layers.Conv2DTranspose):
            ish = l.input.shape
            osh = l.output.shape
            kh, kw = l.kernel_size
            total += 2 * ish[1] * ish[2] * ish[3] * osh[3] * kh * kw
        elif isinstance(l, layers.Conv2D):
            ish = l.input.shape
            osh = l.output.shape
            kh, kw = l.kernel_size
            total += 2 * osh[1] * osh[2] * osh[3] * kh * kw * ish[3]
    return int(total)
