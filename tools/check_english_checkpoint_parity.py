#!/usr/bin/env python3
"""Compare parameter bytes and optimizer files across complete checkpoints.

Metadata/signatures may intentionally differ between runners. No JAX import,
training, GPU initialization or speech-accuracy acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Preserve existing evidence; choose a new output file')
    core=runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    manifests=[core['checkpoint_manifest'](path) for path in (a.reference,a.candidate)]
    if manifests[0]['metadata']['step']!=manifests[1]['metadata']['step']:
        raise ValueError('Expected the same completed global step')
    import numpy as np
    from safetensors import safe_open
    cases=[];count=0
    with safe_open(str(a.reference/'model.safetensors'),framework='np') as reference,safe_open(str(a.candidate/'model.safetensors'),framework='np') as candidate:
        if set(reference.keys())!=set(candidate.keys()):
            raise ValueError('Parameter tree differs')
        for name in sorted(reference.keys()):
            x,y=reference.get_tensor(name),candidate.get_tensor(name)
            if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Invalid parameter comparison: '+name)
            count+=x.size
            cases.append(dict(name=name,shape=list(x.shape),bit_identical=x.tobytes()==y.tobytes()))
    optimizer_equal=core['sha'](a.reference/'optimizer.msgpack')==core['sha'](a.candidate/'optimizer.msgpack')
    result=dict(schema=1,scope=__doc__,step=manifests[0]['metadata']['step'],parameters=count,
                reference_manifest_sha256=core['sha'](a.reference/'manifest.json'),candidate_manifest_sha256=core['sha'](a.candidate/'manifest.json'),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,
                all_parameter_arrays_bit_identical=all(x['bit_identical'] for x in cases),optimizer_bytes_identical=optimizer_equal,
                deployment_permitted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_parameter_arrays_bit_identical'] or not optimizer_equal:
        raise ValueError('Checkpoint state differs; preserve comparison without loosening the test')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':
    main()
