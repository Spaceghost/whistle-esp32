#!/usr/bin/env python3
"""Replay observed context perturbations through the actual C CQ projection.

This diagnoses numeric sensitivity; it does not relax gates or train a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('composition','model','binary','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Preserve prior evidence; choose a fresh output directory')
    import numpy as np
    report_path=a.composition/'decoder.json'
    receipt=json.loads(report_path.read_text())
    export=json.loads((a.model.parent/'export.json').read_text())
    if sha(a.model)!=receipt['model_sha256'] or sha(a.model)!=export['sha256']:
        raise ValueError('Mismatched model')
    if receipt['checkpoint_manifest_sha256']!=export['checkpoint_manifest_sha256']:
        raise ValueError('Changed checkpoint/export pairing')
    trace=receipt['layers']['cross_attention']['composition']
    if (a.composition/'INCOMPLETE').exists() or not receipt['native_probe_logits_bit_identical'] or not trace['original_module_control_passed']:
        raise ValueError('Incomplete or uncontrolled composition trace')
    cfg=export['config']
    count=len(receipt['rows']);width=cfg['dim']
    query_index=1+cfg['dec_layers']*27+9+2+len(cfg['sites'])*4+2+1
    index=query_index+4
    native=np.stack([np.fromfile(a.composition/f'token-{pos}-block-0-cross-gated-context.f32',np.float32).reshape(width)
                     for pos in range(count)])
    retained=np.stack([np.fromfile(a.composition/f'token-{pos}-block-0-cross-projection.f32',np.float32).reshape(width)
                       for pos in range(count)])
    shifted=np.fromfile(a.composition/'composed-cross-gated-context.f32',np.float32).reshape(count,width)
    simulated=np.fromfile(a.composition/'composed-cross-original-projection.f32',np.float32).reshape(count,width)
    if not all(np.isfinite(v).all() for v in (native,retained,shifted,simulated)):
        raise ValueError('Nonfinite replay evidence')
    a.output.mkdir(parents=True)
    (a.output/'INCOMPLETE').write_text('Native context replay incomplete.\n')
    outputs={}
    for name,values in (('native',native),('observed-perturbation',shifted)):
        source=a.output/(name+'-input.f32');target=a.output/(name+'-output.f32');values.tofile(source)
        subprocess.run([str(a.binary.resolve()),str(a.model.resolve()),'cq-linear',str(index),str(source.resolve()),str(target.resolve())],check=True,timeout=60)
        outputs[name]=np.fromfile(target,np.float32).reshape(count,width)
        if not np.isfinite(outputs[name]).all():
            raise ValueError('Nonfinite native output')
    if not np.array_equal(outputs['native'].view(np.uint32),retained.view(np.uint32)):
        raise ValueError('Native projection replay differs from retained baseline; preserve evidence')
    def relative(x,y):
        return float(np.linalg.norm(x.astype(np.float64)-y)/max(float(np.linalg.norm(y.astype(np.float64))),1e-12))
    rows=[]
    for pos in range(count):
        actual=outputs['observed-perturbation'][pos]
        delta=actual.astype(np.float64)-retained[pos]
        simulated_delta=simulated[pos].astype(np.float64)-retained[pos]
        denom=float(np.linalg.norm(delta)*np.linalg.norm(simulated_delta))
        rows.append(dict(position=pos,input_relative_l2_change=relative(shifted[pos],native[pos]),
                         native_output_relative_l2_change=relative(actual,retained[pos]),
                         simulated_vs_perturbed_native_relative_l2_error=relative(simulated[pos],actual),
                         error_vector_cosine=float(delta@simulated_delta/denom) if denom else None))
    result=dict(schema=1,scope=__doc__,model_sha256=sha(a.model),composition_receipt_sha256=sha(report_path),
                native_binary_sha256=sha(a.binary),checker_sha256=sha(Path(__file__)),projection_tensor_index=index,
                native_baseline_bit_identical=True,rows=rows,full_fast_gate_passed=False,deployment_permitted=False)
    (a.output/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'INCOMPLETE').unlink()
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
