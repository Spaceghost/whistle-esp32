"""CQ export using the pinned trainer's JAX float32 arithmetic.

NumPy reductions/matmuls can fall on a different codebook or FP16 norm
boundary. Keep old exports intact; this is a new, explicitly recorded export.
"""


def pack_matrix(export, quantize, matrix, bits, group):
    import jax.numpy as jnp
    import numpy as np
    if bits not in (2, 4) or group != 128:
        raise ValueError('Unsupported matched English CQ policy')
    w = jnp.asarray(matrix, jnp.float32)
    rows, width = w.shape
    pad = (-width) % group
    padded = jnp.pad(w, ((0, 0), (0, pad))) if pad else w
    groups = padded.reshape(rows, -1, group).astype(jnp.float32)
    hadamard = jnp.asarray(quantize._cq_hadamard_np(group))
    codebook = jnp.asarray(quantize._cq_codebook_np(bits, group))
    rotated = groups @ hadamard
    norm = jnp.sqrt(jnp.sum(rotated ** 2, axis=-1, keepdims=True))
    unit = rotated / jnp.maximum(norm, 1e-12)
    nearest = quantize._cq_nearest(unit, codebook)
    indices = np.asarray(jnp.searchsorted(codebook, nearest), np.uint8).reshape(rows, -1)
    norms = np.asarray(norm.astype(jnp.float16))[:, :, 0]
    return export._pack_lsb(indices, bits), norms


def pack_model(export, quantize, params, config, tokenizer, audio, body_bits):
    import numpy as np
    if body_bits not in (2, 4):
        raise ValueError('Unsupported matched English CQ width')
    original = export._q
    def q(name, matrix, bits, group):
        selected = 4 if name == 'embedding' else bits
        matrix = np.asarray(matrix, np.float32)
        packed, norms = pack_matrix(export, quantize, matrix, selected, group)
        return export._Tensor(name, export.CQ, matrix.shape,
                             packed.tobytes() + norms.tobytes(), group, selected)
    export._q = q
    try:
        return export._pack_cact(params, config, body_bits, 128, tokenizer, audio=audio)
    finally:
        export._q = original
