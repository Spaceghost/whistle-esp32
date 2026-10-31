#!/usr/bin/env python3
"""Compare the declared trainer feature profile to the native research frontend."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import wave


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','binary','needle-source','catalog','audio-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Keep earlier evidence; choose a new output')
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    export=json.loads((a.model.parent/'export.json').read_text())
    if sha(a.model)!=export['sha256'] or export['audio_frontend_profile'] not in (2, 3) or (a.model.parent/'INCOMPLETE').exists():
        raise ValueError('Need a complete, verified English research export')
    helper=runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    helper['verify_source'](a.needle_source);whistle,_,_=helper['source_modules'](a.needle_source)
    import jax.numpy as jnp
    import numpy as np
    cfg=whistle.WhistleConfig.from_saved(export['config'])
    catalog=json.loads(a.catalog.read_text());root=a.audio_root.resolve()
    a.output.mkdir(parents=True);(a.output/'INCOMPLETE').write_text('Native/trainer parity not established.\n')
    examples=[]
    for item in catalog['cases']:
        path=(root/item['file']).resolve()
        if path.parent!=root or sha(path)!=item['sha256']:raise ValueError('Changed natural speech fixture')
        with wave.open(str(path),'rb') as f:
            if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,16000):raise ValueError('Wrong audio geometry')
            pcm=array.array('h',f.readframes(f.getnframes()))
        if sys.byteorder!='little':pcm.byteswap()
        examples.append((item['id'],np.asarray(pcm,dtype=np.float32)/32768.,item['sha256']))
    examples.append(('silence-numerical',np.zeros(16000,np.float32),None))
    examples.append(('quiet-noise-numerical',np.random.default_rng(0).normal(0,1e-6,16000).astype(np.float32),None))
    results=[]
    for i,(name,pcm,source_sha) in enumerate(examples):
        audio=a.output/f'{i}.f32';native=a.output/f'{i}-native.f32';audio.write_bytes(pcm.tobytes())
        subprocess.run([str(a.binary.resolve()),str(a.model.resolve()),'mel',str(audio.resolve()),str(native.resolve())],check=True,timeout=60)
        if export['audio_frontend_profile'] == 3:
            import english_features
            expected,length=english_features.features(cfg,jnp.asarray(pcm[None,:]),jnp.asarray([len(pcm)]),whistle.mel_matrix)
        else:
            expected,length=whistle.features(cfg,jnp.asarray(pcm[None,:]),jnp.asarray([len(pcm)]))
        frames=int(length[0]);expected=np.asarray(expected)[0,:frames];actual=np.fromfile(native,np.float32).reshape(frames,cfg.mels)
        error=np.abs(expected-actual)
        record=dict(name=name,source_audio_sha256=source_sha,frames=frames,max_absolute_error=float(error.max()),rms_error=float(np.sqrt(np.mean(error**2))),all_finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all()),within_tolerance=bool(np.allclose(actual,expected,rtol=2e-3,atol=2e-3)))
        expected.tofile(a.output/f'{i}-trainer.f32');results.append(record);print(json.dumps(record),flush=True)
    report={'schema':1,'scope':'Frontend feature parity only; numerical fixtures are not synthetic voice tests','model_sha256':export['sha256'],'native_binary_sha256':sha(a.binary),
            'audio_frontend_profile':export['audio_frontend_profile'],'frontend_only_fixture':export.get('frontend_only_fixture',False),'relative_tolerance':2e-3,'absolute_tolerance':2e-3,'examples':results,'all_passed':all(x['all_finite'] and x['within_tolerance'] for x in results),'deployment_permitted':False}
    (a.output/'frontend.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_passed']:raise ValueError('Native/trainer frontend differs; preserve failure evidence')
    (a.output/'INCOMPLETE').unlink()


if __name__=='__main__':main()
