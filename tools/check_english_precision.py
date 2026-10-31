#!/usr/bin/env python3
"""Check mixed CQ packing against pinned four-bit bytes and two-bit QAT math."""
import argparse,json,os,runpy
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--needle-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Choose a fresh evidence path')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    h=runpy.run_path(str(Path(__file__).with_name('train_english_student.py')));h['verify_source'](a.needle_source)
    _,q,_=h['source_modules'](a.needle_source)
    import importlib
    e=importlib.import_module('_ack_needle.model.export')
    import numpy as np
    import jax
    import jax.numpy as jnp
    import english_precision as ep
    rng=np.random.default_rng(0);results=[]
    for shape in [(7,128),(3,513),(5,512)]:
        matrix=rng.normal(0,.02,shape).astype(np.float32)
        packed,norms=ep.pack_matrix(e,matrix,4,128);old,old_norms=e._cq_pack(matrix,4,128)
        assert packed.tobytes()==old.tobytes() and norms.tobytes()==old_norms.tobytes()
        for bits in (2,4):
            packed,norms=ep.pack_matrix(e,matrix,bits,128)
            restored=e._cq_unpack(packed,norms,*shape,bits,128)
            trained=np.asarray(q.cq_quantize(jnp.asarray(matrix),bits,128))
            error=float(np.max(np.abs(restored-trained)))
            assert np.allclose(restored,trained,atol=2e-6,rtol=2e-5),(shape,bits,error)
            results.append(dict(shape=shape,bits=bits,max_absolute_error=error))
    params={'embedding':{'embedding':jnp.asarray(rng.normal(0,.02,(7,128)).astype(np.float32))},
            'encoder':{'dense':{'kernel':jnp.asarray(rng.normal(0,.02,(128,7)).astype(np.float32))}}}
    before=np.asarray(params['embedding']['embedding']).copy();mixed=ep.qat_params(params,q,2)
    assert np.array_equal(np.asarray(mixed['embedding']['embedding']),np.asarray(q.cq_ste(params['embedding']['embedding'],4)))
    assert np.array_equal(np.asarray(mixed['encoder']['dense']['kernel']),np.asarray(q.cq_ste(params['encoder']['dense']['kernel'].T,2)).T)
    assert np.array_equal(before,np.asarray(params['embedding']['embedding']))
    gradient=jax.grad(lambda tree:sum(jnp.sum(x) for x in jax.tree.leaves(ep.qat_params(tree,q,2))))(params)
    assert all(np.array_equal(np.asarray(x),np.ones(x.shape,np.float32)) for x in jax.tree.leaves(gradient))
    report={'schema':1,'four_bit_bytes_identical':True,'two_and_four_bit_qat_export_parity':results,'mixed_head_body_policy_verified':True,'parameters_not_mutated':True,'fixture_straight_through_gradients_are_identity':True,'scope':'Quantization math only; no speech accuracy or deployment acceptance'}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
