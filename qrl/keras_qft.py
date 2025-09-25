# keras_qft.py
# MIT/Apache-2.0

from typing import Optional, Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class QFTConfigTF:
    def __init__(
        self,
        bands: int = 16,
        aux_loss_weight: float = 0.05,
        entropy_weight: float = 0.1,
        reward_weight: float = 0.2,
        entropy_penalty: float = 0.2,
        normalize_mag: bool = True,
        use_window: bool = True,
        eps: float = 1e-8,
    ):
        self.bands = bands
        self.aux_loss_weight = aux_loss_weight
        self.entropy_weight = entropy_weight
        self.reward_weight = reward_weight
        self.entropy_penalty = entropy_penalty
        self.normalize_mag = normalize_mag
        self.use_window = use_window
        self.eps = eps

def hann_window(T: int, dtype=tf.float32):
    n = tf.range(T, dtype=dtype)
    return 0.5 - 0.5 * tf.cos(2.0 * tf.constant(tf.constant(3.141592653589793*2.0)) * n / tf.cast(T - 1, dtype))

def spectral_entropy(mag, eps=1e-8):
    p = mag / (tf.reduce_sum(mag, axis=-1, keepdims=True) + eps)
    return -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1)

class MotifBankTF(layers.Layer):
    def __init__(self, num_motifs: int, bands: int, learnable: bool = False, init: Optional[tf.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        if init is None:
            init = tf.random.normal([num_motifs, bands], stddev=0.01)
            init = tf.tensor_scatter_nd_update(init, indices=[[0,0]], updates=[1.0])  # bias to low-freq
        self.motifs = self.add_weight(
            name="motifs",
            shape=init.shape,
            initializer=tf.constant_initializer(init.numpy() if hasattr(init, "numpy") else init),
            trainable=learnable,
        )

    def call(self):
        return tf.linalg.l2_normalize(self.motifs, axis=-1)

class QFTTraceRegularizerTF(layers.Layer):
    """
    Call with hidden states H [B,T,D] -> dict with:
      aux_loss (scalar tensor), reward_bonus [B], router_feats [B,F], etc.
    """
    def __init__(self, cfg: QFTConfigTF, motif_bank: MotifBankTF, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.motif_bank = motif_bank

    def _pool(self, H):
        # mean over D to get scalar per token: [B,T]
        return tf.reduce_mean(H, axis=-1)

    def _qft(self, x):
        # x: [B,T]
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        if self.cfg.use_window:
            w = hann_window(int(x.shape[1]))
            x = x * w
        # rFFT over last axis
        F = tf.signal.rfft(x)              # [B, T//2+1]
        mag = tf.abs(F)
        K = tf.minimum(self.cfg.bands, tf.shape(mag)[1])
        mag = mag[:, :K]
        if self.cfg.normalize_mag:
            mag = mag / (tf.reduce_sum(mag, axis=-1, keepdims=True) + self.cfg.eps)
        ent = spectral_entropy(mag, self.cfg.eps)  # [B]
        return mag, ent

    def _motif_scores(self, mag):
        motifs = self.motif_bank()            # [M,K]
        mag_n = tf.linalg.l2_normalize(mag, axis=-1)
        sims = tf.matmul(mag_n, motifs, transpose_b=True)  # [B,M]
        return sims

    def call(self, H: tf.Tensor) -> Dict[str, tf.Tensor]:
        x = self._pool(H)                     # [B,T]
        mag, ent = self._qft(x)               # [B,K], [B]
        sims = self._motif_scores(mag)        # [B,M]
        best_sim = tf.reduce_max(sims, axis=-1)  # [B]

        L_fit = tf.reduce_mean(1.0 - best_sim)   # simple fit loss
        L_ent = tf.reduce_mean(ent)
        aux_loss = self.cfg.aux_loss_weight * (L_fit + self.cfg.entropy_weight * L_ent)

        r_qft = best_sim - self.cfg.entropy_penalty * ent
        reward_bonus = self.cfg.reward_weight * tf.stop_gradient(r_qft)

        router_feats = tf.concat([tf.expand_dims(ent, -1),
                                  tf.expand_dims(best_sim, -1),
                                  sims], axis=-1)

        # Expose aux loss to Keras if used inside a Model
        self.add_loss(aux_loss)

        return {
            "aux_loss": aux_loss,
            "reward_bonus": reward_bonus,
            "router_feats": router_feats,
            "qft_mag": mag,
            "qft_entropy": ent,
            "motif_sims": sims,
        }
