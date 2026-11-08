"""Combo loss (Equation 3): L = 0.3 BCE + 0.4 Dice + 0.3 focal (gamma = 2, alpha = 0.25).

The `full_nofocal` ablation drops the focal term and keeps the other two weights
unchanged (0.3 BCE + 0.4 Dice), so that only the focal term differs.
"""


def make_losses(tcfg, variant):
    import tensorflow as tf
    w = tcfg["loss_w"]
    g, a = tcfg["focal_gamma"], tcfg["focal_alpha"]
    use_focal = variant != "full_nofocal"
    eps = 1e-7

    def _prep(y, p):
        return tf.cast(y, tf.float32), tf.clip_by_value(tf.cast(p, tf.float32), eps, 1.0 - eps)

    def bce(y, p):
        y, p = _prep(y, p)
        return tf.reduce_mean(-(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p)))

    def dice_loss(y, p):
        y, p = _prep(y, p)
        inter = tf.reduce_sum(y * p)
        return 1.0 - (2.0 * inter + 1.0) / (tf.reduce_sum(y) + tf.reduce_sum(p) + 1.0)

    def focal(y, p):
        y, p = _prep(y, p)
        pt = y * p + (1.0 - y) * (1.0 - p)
        at = y * a + (1.0 - y) * (1.0 - a)
        return tf.reduce_mean(-at * tf.pow(1.0 - pt, g) * tf.math.log(pt))

    def combo_loss(y, p):
        loss = w["bce"] * bce(y, p) + w["dice"] * dice_loss(y, p)
        if use_focal:
            loss = loss + w["focal"] * focal(y, p)
        return loss

    def dice_coef(y, p):
        """Soft Dice of a batch (the Keras metric plotted in Figure 4)."""
        y, p = tf.cast(y, tf.float32), tf.cast(p, tf.float32)
        inter = tf.reduce_sum(y * p)
        return (2.0 * inter + 1e-6) / (tf.reduce_sum(y) + tf.reduce_sum(p) + 1e-6)

    return combo_loss, dice_coef
