"""Compare composed normalization/cross-attention and an explicit JAX barrier.

The barrier is a diagnostic candidate, never a changed training policy or gate.
"""


def compare(a, cfg, whistle, block, kwargs, trace_cross, post_self, native_projection):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    import english_native_quant as nq
    quant = not a.floating
    class Composed(nn.Module):
        capture: bool = False
        barrier: bool = False

        @nn.compact
        def __call__(self, x, mem, mem_mask):
            x = whistle.ZCRMSNorm(dtype=cfg.jax_dtype, name='cross_norm')(x)
            if self.capture:
                self.sow('intermediates', 'normalized', x)
            if self.barrier:
                x = jax.lax.optimization_barrier(x)
            cross = trace_cross if self.capture else whistle.CrossAttention
            return cross(cfg, cfg.jax_dtype, name='cross_attn')(x, mem, mem_mask, quant)

    params = {'cross_norm':block['cross_norm'],'cross_attn':block['cross_attn']}
    inputs = jnp.asarray(post_self)
    arguments = (kwargs['mem'], kwargs['mem_mask'])
    results = {}
    captures = {}
    for name,capture,barrier in (('original',False,False),('captured',True,False),('barrier-after-norm',False,True)):
        module = Composed(capture=capture,barrier=barrier)
        if capture:
            out,values = jax.jit(lambda x:module.apply({'params':params},x,*arguments,mutable=['intermediates']))(inputs)
            captures = values['intermediates']
        else:
            out = jax.jit(lambda x:module.apply({'params':params},x,*arguments))(inputs)
        results[name] = np.asarray(out)
        results[name].tofile(a.output/f'composed-cross-{name}-projection.f32')
    controls = [float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-12))
                for x,y in zip(results['captured'][0],results['original'][0])]
    rows=[]
    for name,out in results.items():
        if out.shape!=native_projection.shape or not np.isfinite(out).all():
            raise ValueError('Invalid composed cross-attention output')
        for pos in range(out.shape[1]):
            relative=float(np.linalg.norm(out[0,pos]-native_projection[0,pos])/max(float(np.linalg.norm(out[0,pos])),1e-12))
            rows.append(dict(mode=name,position=pos,relative_l2_error=relative,
                             maximum_absolute_error=float(np.max(np.abs(out[0,pos]-native_projection[0,pos]))),
                             within_existing_threshold=relative<2e-4))
    normalized=np.asarray(captures['normalized'][0])
    context=np.asarray(captures['cross_attn']['gated-context'][0])
    normalized.tofile(a.output/'composed-cross-normalized.f32')
    context.tofile(a.output/'composed-cross-gated-context.f32')
    code_rows=[]
    if quant:
        def codes(x):
            groups=x.reshape(*x.shape[:-1],-1,128)
            h=nq.hadamard(groups)
            maximum=jnp.max(jnp.abs(h),axis=-1,keepdims=True)
            scale=jnp.where(maximum>0,maximum/jnp.float32(127),jnp.float32(1))
            return nq.round_away(h*(jnp.float32(1)/scale)).reshape(x.shape)
        values=np.asarray(jax.jit(codes)(jnp.asarray(context)))
        native=np.stack([np.fromfile(a.output/f'token-{p}-block-0-cross-context-q8-codes.f32',np.float32)
                         for p in range(context.shape[1])])[None,:]
        for pos in range(context.shape[1]):
            code_rows.append(dict(position=pos,context_codes_different_from_actual_kernel=int(np.count_nonzero(values[0,pos]!=native[0,pos]))))
    return dict(scope=__doc__,rows=rows,context_q8_codes=code_rows,
                captured_vs_original_per_token_relative_l2_error=controls,original_module_control_tolerance=2e-6,
                original_module_control_passed=all(np.isfinite(v) and v<2e-6 for v in controls),
                barrier_is_experimental=True,training_policy_changed=False,native_threshold_unchanged=2e-4,
                deployment_permitted=False)
