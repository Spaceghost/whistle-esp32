"""Cross-attention internals using native normalized input and kernel Q8 codes."""
from functools import partial
import subprocess


def compare(a, cfg, whistle, block, kwargs):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    import english_native_quant as nq
    if cfg.flash:
        raise ValueError('This diagnostic covers the non-flash attention path only')
    quant = not a.floating
    class TraceCross(whistle.CrossAttention):
        @nn.compact
        def __call__(self, x, mem, mem_mask, quant=False):
            c, dt = self.cfg, self.dtype
            B = x.shape[0]
            x = whistle._aq(x, quant)
            dense = partial(nn.Dense, dtype=dt, use_bias=False, kernel_init=whistle.default_init())
            q = dense(c.heads*c.qk_dim, name='q_proj')(x).reshape(B,-1,c.heads,c.qk_dim)
            self.sow('intermediates', 'raw-query', q.reshape(B,-1,c.heads*c.qk_dim))
            k = dense(c.heads*c.qk_dim, name='k_proj')(mem).reshape(B,-1,c.heads,c.qk_dim)
            v = dense(c.heads*c.v_dim, name='v_proj')(mem).reshape(B,-1,c.heads,c.v_dim)
            q = whistle.ZCRMSNorm(dtype=dt, name='q_norm')(q.transpose(0,2,1,3))
            k = whistle.ZCRMSNorm(dtype=dt, name='k_norm')(k.transpose(0,2,1,3))
            v = v.transpose(0,2,1,3)
            q = whistle._quantize.maybe_quant_query(q, quant)
            k = whistle._quantize.maybe_quant_kv(k, quant)
            v = whistle._quantize.maybe_quant_kv(v, quant)
            self.sow('intermediates', 'normalized-query', q.transpose(0,2,1,3).reshape(B,-1,c.heads*c.qk_dim))
            self.sow('intermediates', 'cached-key', k.transpose(0,2,1,3).reshape(B,-1,c.heads*c.qk_dim))
            self.sow('intermediates', 'cached-value', v.transpose(0,2,1,3).reshape(B,-1,c.heads*c.v_dim))
            out = whistle._cross_attend(q.transpose(0,2,1,3), k.transpose(0,2,1,3), v.transpose(0,2,1,3), mem_mask)
            out = out.reshape(B,-1,c.heads*c.v_dim)
            gate = dense(c.heads*c.v_dim, name='gate_proj')(x)
            self.sow('intermediates', 'gate-logits', gate)
            out = out*nn.sigmoid(gate)
            self.sow('intermediates', 'gated-context', out)
            out = whistle._aq(out, quant)
            out = nn.Dense(c.dim, dtype=dt, use_bias=False, kernel_init=whistle.residual_init(c.dec_layers), name='out_proj')(out)
            self.sow('intermediates', 'projection', out)
            return out

    count = kwargs['mask'].shape[-1]
    def read(stage, width):
        return np.stack([np.fromfile(a.output/f'token-{pos}-block-0-cross-{stage}.f32',np.float32).reshape(width)
                         for pos in range(count)])[None,:]
    inputs = jnp.asarray(read('normalized-input', cfg.dim))
    arguments = dict(mem=kwargs['mem'], mem_mask=kwargs['mem_mask'], quant=quant)
    original = whistle.CrossAttention(cfg, cfg.jax_dtype)
    reference = np.asarray(jax.jit(lambda x: original.apply({'params':block['cross_attn']},x,**arguments))(inputs))
    module = TraceCross(cfg,cfg.jax_dtype)
    actual, captured = jax.jit(lambda x: module.apply({'params':block['cross_attn']},x,**arguments,
                                                    mutable=['intermediates']))(inputs)
    values = {name:np.asarray(value[0]) for name,value in captured['intermediates'].items()}
    controls = [float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-12)) for x,y in zip(np.asarray(actual)[0], reference[0])]
    native = {stage:read(stage,width) for stage,width in (('raw-query',cfg.heads*cfg.qk_dim),
              ('gate-logits',cfg.heads*cfg.v_dim),('gated-context',cfg.heads*cfg.v_dim),('projection',cfg.dim))}
    native['normalized-query'] = np.stack([np.concatenate([np.fromfile(a.output/f'token-{pos}-block-0-cross-query-head-{head}.f32',np.float32)
                                                         for head in range(cfg.heads)]) for pos in range(count)])[None,:]
    frames = kwargs['mem'].shape[1]
    for name,filename,width in (('cached-key','cross-k.f32',cfg.qk_dim),('cached-value','cross-v.f32',cfg.v_dim)):
        raw = np.fromfile(a.output/filename,np.float32).reshape(cfg.dec_layers,cfg.heads,frames,width)[0]
        native[name] = raw.transpose(1,0,2).reshape(1,frames,cfg.heads*width)
    rows = []
    def record(stage,value,target,mode='cumulative'):
        if value.shape != target.shape or not np.isfinite(value).all() or not np.isfinite(target).all():
            raise ValueError('Invalid cross-attention geometry or values')
        value.tofile(a.output/f'cross-{stage}-{mode}-jax.f32')
        for pos in range(value.shape[1]):
            relative = float(np.linalg.norm(value[0,pos]-target[0,pos])/max(float(np.linalg.norm(value[0,pos])),1e-12))
            rows.append(dict(stage=stage,mode=mode,position=pos,position_kind='memory-frame' if stage.startswith('cached-') else 'decoder-token',
                             relative_l2_error=relative,maximum_absolute_error=float(np.max(np.abs(value[0,pos]-target[0,pos]))),
                             within_existing_threshold=relative < 2e-4))
    for name,value in values.items():
        record(name,value,native[name])
    head = block['cross_attn']['out_proj']['kernel']
    project = jax.jit(lambda x:whistle._aq(x,quant)@head)
    shared = np.asarray(project(jnp.asarray(native['gated-context'])))
    record('projection',shared,native['projection'],'shared-context')
    code_rows = []
    if quant:
        def prepare(x):
            groups = x.reshape(*x.shape[:-1],-1,128)
            rotated = nq.hadamard(groups)
            maximum = jnp.max(jnp.abs(rotated),axis=-1,keepdims=True)
            scale = jnp.where(maximum>0,maximum/jnp.float32(127),jnp.float32(1))
            return nq.round_away(rotated*(jnp.float32(1)/scale)).reshape(x.shape),scale[...,0]
        prepare = jax.jit(prepare)
        own_codes,_ = prepare(jnp.asarray(values['gated-context']))
        shared_codes,shared_scales = prepare(jnp.asarray(native['gated-context']))
        native_codes = read('context-q8-codes',cfg.heads*cfg.v_dim)
        native_scales = read('context-q8-scales',cfg.heads*cfg.v_dim//128)
        for pos in range(count):
            code_rows.append(dict(position=pos,own_context_codes_different_from_actual_kernel=int(np.count_nonzero(np.asarray(own_codes)[0,pos]!=native_codes[0,pos])),
                                  shared_native_context_codes_different_from_actual_kernel=int(np.count_nonzero(np.asarray(shared_codes)[0,pos]!=native_codes[0,pos])),
                                  shared_scale_max_absolute_error=float(np.max(np.abs(np.asarray(shared_scales)[0,pos]-native_scales[0,pos])))))
    norm = whistle.ZCRMSNorm(dtype=cfg.jax_dtype)
    post_self = np.stack([np.fromfile(a.output/f'token-{pos}-block-0-post-self.f32',np.float32).reshape(cfg.dim)
                          for pos in range(count)])[None,:]
    normalized = np.asarray(jax.jit(lambda x:norm.apply({'params':block['cross_norm']},x))(jnp.asarray(post_self)))
    norm_rows = []
    if quant:
        # Current verified layout: embedding, decoder blocks (27 records each),
        # nine mHC tensors, two permutations, four/site engram tensors, final
        # norm, audio manifest, then cross-attention Q at offset one. Refuse a
        # wrong index through exact replay against the captured raw queries.
        query_index = 1+cfg.dec_layers*27+9+2+len(cfg.sites)*4+2+1
        replays = {}
        for name,value in (('native-normalized',np.asarray(inputs)),('jax-normalized',normalized)):
            source = a.output/f'query-replay-{name}-input.f32'
            target = a.output/f'query-replay-{name}-output.f32'
            value.tofile(source)
            subprocess.run([str(a.linear_binary.resolve()),str(a.model.resolve()),'cq-linear',str(query_index),str(source.resolve()),str(target.resolve())],check=True,timeout=60)
            replays[name] = np.fromfile(target,np.float32).reshape(native['raw-query'].shape)
        if not np.array_equal(replays['native-normalized'].view(np.uint32),native['raw-query'].view(np.uint32)):
            raise ValueError('Query replay does not reproduce captured native projection; no sensitivity attribution')
        for pos in range(count):
            source = np.asarray(inputs)[0,pos]
            baseline = replays['native-normalized'][0,pos]
            shifted = replays['jax-normalized'][0,pos]
            norm_rows.append(dict(position=pos,normalized_input_relative_l2_change=float(np.linalg.norm(source-normalized[0,pos])/max(float(np.linalg.norm(source)),1e-12)),
                                  native_query_relative_l2_change=float(np.linalg.norm(baseline-shifted)/max(float(np.linalg.norm(baseline)),1e-12)),
                                  native_query_max_absolute_change=float(np.max(np.abs(baseline-shifted)))))
    composition = None
    if a.cross_composition:
        import english_cross_composition
        composition = english_cross_composition.compare(a,cfg,whistle,block,kwargs,TraceCross,post_self,native['projection'])
    return dict(scope=__doc__,rows=rows,q8_codes=code_rows,normalization_replay=norm_rows,composition=composition,
                query_replay_baseline_bit_identical=bool(quant),
                traced_vs_original_per_token_relative_l2_error=controls,
                original_module_control_tolerance=2e-6,original_module_control_passed=all(np.isfinite(v) and v<2e-6 for v in controls),
                native_threshold_unchanged=2e-4,deployment_permitted=False)
