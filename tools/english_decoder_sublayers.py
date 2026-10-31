"""First decoder block boundaries, retaining original-module controls."""
import math


def compare(a, cfg, whistle, block, kwargs, flags):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    quant = not a.floating
    # Match the pinned DecoderBlock, adding only captured values and optional
    # identical-native-state injection for an isolated downstream operation.
    class TraceBlock(whistle.DecoderBlock):
        replacement: str = ''

        @nn.compact
        def __call__(self, x, replacement_value, mask=None, rope=None, quant=False,
                     engram_kv=None, site_flags=None, mem=None, mem_mask=None):
            c, dt = self.cfg, self.dtype
            self.sow('intermediates', 'input', x)
            if engram_kv is not None:
                ek, ev = engram_kv
                alpha = nn.sigmoid(jnp.einsum('btd,sbtd->sbt', whistle._rms_unit(x), whistle._rms_unit(ek))/math.sqrt(c.dim))
                x = x+jnp.einsum('s,sbt,sbtd->btd', site_flags.astype(jnp.float32), alpha, ev.astype(jnp.float32)).astype(x.dtype)
            self.sow('intermediates', 'post-engram', x)
            skip = x
            x = whistle.ZCRMSNorm(dtype=dt)(x)
            x = whistle.MultiHeadAttention(c.heads, c.kv_heads, c.dim, c.dec_layers, dt, c.flash,
                    qk_head_dim=c.qk_dim, v_head_dim=c.v_dim, qkv_conv_taps=c.taps,
                    name='self_attn')(x, mask=mask, rope=rope, quant=quant)
            x = whistle.ZCRMSNorm(dtype=dt, name='post_attn_norm')(x)
            x = skip+self._gate('attn_gate')*x
            self.sow('intermediates', 'post-self', x)
            if self.replacement == 'post-self':
                x = replacement_value
            skip = x
            x = whistle.ZCRMSNorm(dtype=dt, name='cross_norm')(x)
            x = whistle.CrossAttention(c, dt, name='cross_attn')(x, mem, mem_mask, quant)
            x = whistle.ZCRMSNorm(dtype=dt, name='post_cross_norm')(x)
            x = skip+self._gate('cross_gate')*x
            self.sow('intermediates', 'post-cross', x)
            if self.replacement == 'post-cross':
                x = replacement_value
            skip = x
            x = whistle.ZCRMSNorm(dtype=dt, name='pre_hada_norm')(x)
            x = skip+whistle.HadamardMLP(c.dim, dt, name='hadamard_mlp')(x)
            self.sow('intermediates', 'post-mlp', x)
            return x

    count = kwargs['mask'].shape[-1]
    def read(stage):
        return np.stack([np.fromfile(a.output/f'token-{pos}-block-0-{stage}.f32', np.float32).reshape(cfg.dim)
                         for pos in range(count)])[None, :]
    native = {stage: read(stage) for stage in ('input', 'post-engram', 'post-self', 'post-cross', 'post-mlp')}
    arguments = dict(kwargs, site_flags=flags, quant=quant)
    inputs = jnp.asarray(native['input'])
    original = whistle.DecoderBlock(cfg, cfg.jax_dtype)
    reference = np.asarray(jax.jit(lambda x: original.apply({'params': block}, x, **arguments))(inputs))
    def run(replacement=''):
        module = TraceBlock(cfg, cfg.jax_dtype, replacement=replacement)
        supplied = jnp.asarray(native[replacement] if replacement else native['input'])
        out, values = jax.jit(lambda x, v: module.apply({'params': block}, x, v, **arguments,
                                                     mutable=['intermediates']))(inputs, supplied)
        return np.asarray(out), {name: np.asarray(value[0]) for name, value in values['intermediates'].items()}
    traced, cumulative = run()
    controls = [float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)), 1e-12)) for x, y in zip(traced[0], reference[0])]
    rows = []
    def record(stage, mode, value):
        target = native[stage]
        if value.shape != target.shape or not np.isfinite(value).all() or not np.isfinite(target).all():
            raise ValueError('Invalid sublayer geometry or values')
        value.tofile(a.output/f'block-0-{stage}-{mode}-jax.f32')
        for pos in range(count):
            relative = float(np.linalg.norm(value[0,pos]-target[0,pos])/max(float(np.linalg.norm(value[0,pos])), 1e-12))
            rows.append(dict(stage=stage, mode=mode, position=pos, relative_l2_error=relative,
                             maximum_absolute_error=float(np.max(np.abs(value[0,pos]-target[0,pos]))),
                             within_existing_threshold=relative < 2e-4))
    for stage, value in cumulative.items():
        record(stage, 'cumulative', value)
    record('post-self', 'shared-input', cumulative['post-self'])
    for boundary, stage in (('post-self','post-cross'), ('post-cross','post-mlp')):
        _, isolated = run(boundary)
        record(stage, 'shared-input', isolated[stage])
    return dict(scope='First decoder layer only; native block input and isolated sublayer boundaries', rows=rows,
                traced_vs_original_per_token_relative_l2_error=controls, original_module_control_tolerance=2e-6,
                original_module_control_passed=all(np.isfinite(v) and v < 2e-6 for v in controls),
                native_threshold_unchanged=2e-4, deployment_permitted=False)
