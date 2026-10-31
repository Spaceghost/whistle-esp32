"""Differentiable simulation of this fork's fast CQ kernel, not upstream A8.

The C path rotates 128-value activation groups, rounds half away from zero,
uses signed-eight-bit codebook levels, and retains float attention KV caches.
Embedding lookup uses original CQ values; the tied output matmul uses the
kernel's rounded codebook. All changes to pinned modules are in memory only.
"""
POLICY = 'cq-h128-a8-away-cb8-kv-f32-v1'
OUTPUT_HEAD = 'ack_native_output_embedding'


def hadamard(x):
    """Normalized length-128 FWHT, in the C butterfly operation order."""
    import jax.numpy as jnp
    if x.shape[-1] != 128:
        raise ValueError('Expected a 128-value group')
    shape = x.shape
    for width in (1, 2, 4, 8, 16, 32, 64):
        pairs = x.reshape(*shape[:-1], -1, 2, width)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        x = jnp.concatenate((a + b, a - b), axis=-1).reshape(shape)
    return x * jnp.float32(0.08838834764831845)


def round_away(x):
    import jax.numpy as jnp
    return jnp.trunc(x + jnp.where(x >= 0, jnp.float32(.5), jnp.float32(-.5)))


def activation_values(x):
    import jax.numpy as jnp
    width = x.shape[-1]
    padded = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, (-width) % 128)])
    groups = padded.reshape(*padded.shape[:-1], -1, 128).astype(jnp.float32)
    rotated = hadamard(groups)
    maximum = jnp.max(jnp.abs(rotated), axis=-1, keepdims=True)
    scale = jnp.where(maximum > 0, maximum / jnp.float32(127), jnp.float32(1))
    integers = round_away(rotated * (jnp.float32(1) / scale))
    return hadamard(integers * scale).reshape(padded.shape)[..., :width].astype(x.dtype)


def activation(x):
    import jax
    return x + jax.lax.stop_gradient(activation_values(x) - x)


def cq_values(w, quantize, bits):
    """Original CQ index/norm decisions, with native GEMM codebook levels."""
    import jax.numpy as jnp
    cb = jnp.asarray(quantize._cq_codebook_np(bits, 128))
    H = jnp.asarray(quantize._cq_hadamard_np(128))
    width = w.shape[-1]
    padded = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, (-width) % 128)])
    groups = padded.reshape(*padded.shape[:-1], -1, 128).astype(jnp.float32)
    rotated = groups @ H
    norm = jnp.sqrt(jnp.sum(rotated ** 2, axis=-1, keepdims=True))
    nearest = quantize._cq_nearest(rotated / jnp.maximum(norm, 1e-12), cb)
    sw = jnp.float32(127) / jnp.max(jnp.abs(cb))
    # Index selection still uses the original CQ codebook. Only the linear
    # kernel's effective levels are rounded to int8, matching nd_cq_q8_gemm.
    level = round_away(nearest * sw)
    effective = level * (norm.astype(jnp.float16).astype(jnp.float32) * (jnp.float32(1) / sw))
    return (effective @ H).reshape(padded.shape)[..., :width].astype(w.dtype)


def cq_ste(w, quantize, bits):
    import jax
    return w + jax.lax.stop_gradient(cq_values(w, quantize, bits) - w)


def parameters(params, quantize, body_bits):
    """Derive deploy values without adding a saved/trainable parameter."""
    import jax
    import jax.numpy as jnp
    if body_bits not in (2, 4) or 'ab_scales' in params or OUTPUT_HEAD in params:
        raise ValueError('Unsupported native training policy')
    def convert(path, leaf):
        if not quantize._is_quant_leaf(path, leaf):
            return leaf
        names = tuple(str(entry.key) for entry in path if hasattr(entry, 'key'))
        if names[-1] == 'embedding':
            bits = 4 if names == ('embedding', 'embedding') else body_bits
            return quantize.cq_ste(leaf, bits)  # Original CQ for table lookup.
        if quantize._reduces_second_last(path):
            return jnp.swapaxes(cq_ste(jnp.swapaxes(leaf, -1, -2), quantize, body_bits), -1, -2)
        return cq_ste(leaf, quantize, body_bits)
    result = jax.tree_util.tree_map_with_path(convert, params)
    result[OUTPUT_HEAD] = cq_ste(params['embedding']['embedding'], quantize, 4)
    return result


def install(whistle, quantize):
    """Install a research simulation; return a restoration callback."""
    import importlib
    architecture = importlib.import_module(whistle.__package__ + '.architecture')
    originals = (architecture.fake_quant_act, quantize.maybe_quant_query,
                 quantize.maybe_quant_kv, whistle.Whistle.logits, whistle.Stem)
    architecture.fake_quant_act = activation
    quantize.maybe_quant_query = lambda x, quant: x
    quantize.maybe_quant_kv = lambda x, quant: x
    # Native CQ linear dispatch also quantizes the two stem pointwise
    # projections. The pinned trainer only quantizes the final stem output.
    # Subclassing lets Flax wrap @compact normally and preserves param names.
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    class NativeStem(whistle.Stem):
        @nn.compact
        def __call__(self, x, lengths, quant):
            c = self.cfg
            def down(x, w, lengths):
                x = jax.lax.conv_general_dilated(
                    x, w.astype(x.dtype), (2, 2), ((1, 1), (1, 1)),
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=x.shape[-1])
                lengths = (lengths + 1) // 2
                return x * (jnp.arange(x.shape[1]) < lengths[:, None])[..., None, None], lengths
            x, lengths = down(x[..., None], self.param('w', whistle.default_init(), (3, 3, 1, c.stem)), lengths)
            for i in range(1, c.stem_stages):
                x, lengths = down(jax.nn.silu(x), self.param(f'dw_{i}', whistle.default_init(), (3, 3, 1, c.stem)), lengths)
                x = nn.Dense(c.stem, dtype=self.dtype, use_bias=False,
                             kernel_init=whistle.default_init(), name=f'pw_{i}')(architecture._aq(x, quant))
            x = jnp.swapaxes(jax.nn.silu(x), 2, 3)
            x = architecture._aq(x.reshape(*x.shape[:2], -1), quant)
            return nn.Dense(c.dim, dtype=self.dtype, use_bias=False,
                            kernel_init=whistle.default_init(), name='out')(x), lengths
    whistle.Stem = NativeStem
    original_logits = whistle.Whistle.logits
    def logits(self, tokens, enc, frames, quant=False):
        if not quant:
            return original_logits(self, tokens, enc, frames, quant)
        import jax.numpy as jnp
        x = self.decode(tokens, enc, frames, quant)
        head = self.variables['params'][OUTPUT_HEAD]
        return activation(x).astype(jnp.float32) @ head.T.astype(jnp.float32)
    whistle.Whistle.logits = logits
    def restore():
        (architecture.fake_quant_act, quantize.maybe_quant_query,
         quantize.maybe_quant_kv, whistle.Whistle.logits, whistle.Stem) = originals
    return restore
