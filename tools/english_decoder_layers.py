"""Host decoder layer localization; no production hooks or parameter changes."""
from functools import partial


def compare(a, model, params, cfg, whistle, tokens, enc, full_hidden):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    quant = not a.floating
    tokens = jnp.asarray(tokens[None, :], dtype=jnp.int32)
    mask = whistle.make_causal_mask(tokens.shape[1])
    def prepare(m, tok, memory):
        pe = nn.sigmoid(m.pe_gate).astype(memory.dtype) * whistle.sinusoids(cfg, memory.shape[1]).astype(memory.dtype)
        mem = whistle._aq(memory + pe, quant)
        x = m.embedding(tok) * np.sqrt(cfg.dim)
        orders, heads, _ = whistle.engram_geometry(cfg.needle())
        indices = whistle.engram_indices(tok, orders, heads, cfg.slots, 0)
        ngram_ok = jnp.stack([whistle._mask_diag(mask, o-1) for o in orders for _ in range(heads)], axis=-1)
        tap_ok = jnp.stack([whistle._mask_diag(mask, j*max(orders)) for j in range(whistle.ENGRAM_CONV_TAPS)])
        pairs = [e(indices, ngram_ok, tap_ok, quant=quant) for e in m.engrams]
        kv = (jnp.stack([k for k, _ in pairs]), jnp.stack([v for _, v in pairs])) if pairs else None
        return x, mem, kv
    x, mem, kv = model.apply({'params': params}, tokens, jnp.asarray(enc), method=prepare)
    x = jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], cfg.lanes, cfg.dim))
    kwargs = dict(mask=mask, rope=whistle.precompute_rope_freqs(cfg.qk_dim, tokens.shape[1], cfg.rope),
                  engram_kv=kv, mem=mem, mem_mask=jnp.ones(enc.shape[:2], dtype=bool))
    body = whistle._LaneBody(partial(whistle.DecoderBlock, cfg, cfg.jax_dtype), cfg.jax_dtype)
    stack = params['stack']
    records = []
    sublayers = None
    def record(layer, mode, expected, actual):
        expected = np.asarray(expected)[0]
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('Invalid decoder layer geometry or values')
        expected.tofile(a.output / f'layer-{layer}-{mode}-jax.f32')
        for pos in range(expected.shape[0]):
            relative = float(np.linalg.norm(actual[pos]-expected[pos]) / max(float(np.linalg.norm(expected[pos])), 1e-12))
            records.append(dict(layer=layer, mode=mode, position=pos, relative_l2_error=relative,
                                maximum_absolute_error=float(np.max(np.abs(actual[pos]-expected[pos]))),
                                within_existing_threshold=relative < 2e-4))
    for layer in range(cfg.dec_layers):
        block = jax.tree.map(lambda value: value[layer], stack['layers']['block'])
        hc = {name: stack['mhc_'+name][layer] for name in ('phi_pre','phi_post','phi_res','a_pre','a_post','a_res','b_pre','b_post','b_res')}
        lane = jnp.eye(cfg.lanes)[layer % cfg.lanes]
        hc.update(pre_off=8*lane-4, post_off=-4*(1-lane))
        flags = jnp.asarray([float(layer == site) for site in cfg.sites]) if kv is not None else None
        execute = jax.jit(lambda p, value, control: body.apply({'params': {'block': p}}, value, (flags, control), quant, kwargs)[0])
        x = execute(block, x, hc)
        def read(suffix):
            return np.stack([np.fromfile(a.output / f'token-{pos}-{suffix}.f32', np.float32).reshape(cfg.lanes, cfg.dim)
                             for pos in range(tokens.shape[1])])
        actual = read(f'layer-{layer}')
        record(layer, 'cumulative', x, actual)
        native_input = read('init' if layer == 0 else f'layer-{layer-1}')
        isolated = execute(block, jnp.asarray(native_input[None, :]), hc)
        record(layer, 'shared-input', isolated, actual)
        if layer == 0 and a.sublayers:
            import english_decoder_sublayers
            sublayers = english_decoder_sublayers.compare(a, cfg, whistle, block, kwargs, flags)
    pooled = jnp.mean(x, axis=2)
    final = (1+stack['final_norm']['scale']) * pooled / jnp.sqrt(jnp.mean(pooled**2, axis=-1, keepdims=True)+1e-6)
    final = np.asarray(final)
    control = float(np.linalg.norm(final-full_hidden) / max(float(np.linalg.norm(full_hidden)), 1e-12))
    per_token_control = [float(np.linalg.norm(a-b) / max(float(np.linalg.norm(b)), 1e-12))
                         for a, b in zip(final[0], full_hidden[0])]
    final.tofile(a.output / 'layer-trace-final-hidden.f32')
    return dict(scope='Decoder layer boundaries on retained native encoder memory; fixed forced tokens, not speech accuracy',
                sublayers=sublayers,
                rows=records, traced_vs_original_jax_hidden_relative_l2_error=control,
                original_jax_control_within_existing_threshold=bool(np.isfinite(control) and control < 2e-4),
                traced_vs_original_jax_per_token_relative_l2_error=per_token_control,
                original_jax_control_tolerance=2e-6,
                original_jax_control_passed=all(np.isfinite(v) and v < 2e-6 for v in per_token_control),
                relative_tolerance_unchanged=2e-4, deployment_permitted=False)
