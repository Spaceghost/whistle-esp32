"""Stable English frontend profile 3; mel coefficients stay pinned to Needle.

log(P + epsilon) differs from log1p(P / epsilon) by the constant
log(epsilon), which channel mean subtraction removes. Keeping that constant
out of float32 avoids amplifying rounding error in silence and quiet speech.
"""
PROFILE = 3


def features(cfg, wav, lengths, mel_matrix):
    import jax
    import jax.numpy as jnp
    import numpy as np
    wav = wav.astype(jnp.float32) * (jnp.arange(wav.shape[1]) < lengths[:, None])
    window = np.hanning(cfg.win).astype(np.float32)
    _, _, z = jax.scipy.signal.stft(wav, window=window, nperseg=cfg.win,
        noverlap=cfg.win - cfg.hop, nfft=cfg.fft, boundary='zeros', padded=False)
    power = jnp.swapaxes(jnp.abs(z * window.sum()) ** 2, 1, 2)
    x = jnp.log1p((power @ mel_matrix(cfg)) / (2.0 ** -24))
    lengths = lengths // cfg.hop
    mask = (jnp.arange(x.shape[1]) < lengths[:, None])[..., None]
    n = jnp.maximum(lengths, 1)[:, None, None]
    anchor = x[:, :1, :]
    mean = anchor + ((x - anchor) * mask).sum(1, keepdims=True) / n
    std = jnp.sqrt((((x - mean) * mask) ** 2).sum(1, keepdims=True) / jnp.maximum(n - 1, 1))
    return (x - mean) / (std + 1e-5) * mask, lengths


def install(whistle):
    # Override in memory only. Never edit the hash-verified upstream source.
    original = whistle.features
    def configured(cfg, wav, lengths):
        return features(cfg, wav, lengths, whistle.mel_matrix)
    whistle.features = configured
    return original
