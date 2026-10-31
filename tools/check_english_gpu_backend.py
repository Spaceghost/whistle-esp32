#!/usr/bin/env python3
"""Guard-bound single-GPU numeric smoke; no Whistle training or device I/O."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time


def validate_context(record,env,pid,parent):
    if (record.get('state')!='running' or record.get('admitted') is not True or record.get('child_pid')!=pid or
        record.get('guard_pid')!=parent or env.get('ACK_ENGLISH_GPU_GUARD_PID')!=str(parent)):
        raise ValueError('Require a running owned GPU guard')
    visible=env.get('CUDA_VISIBLE_DEVICES','')
    if not visible.startswith('GPU-') or ',' in visible or hashlib.sha256(visible.encode()).hexdigest()[:16]!=record['device_fingerprint']:
        raise ValueError('Selected GPU does not match the guard')
    if env.get('JAX_PLATFORMS')!='cuda' or env.get('XLA_PYTHON_CLIENT_PREALLOCATE')!='true' or env.get('JAX_SKIP_CUDA_CONSTRAINTS_CHECK'):
        raise ValueError('GPU backend or allocation/version checks differ from the guard')
    fraction=float(env['XLA_PYTHON_CLIENT_MEM_FRACTION'])
    if not math.isfinite(fraction) or not 0<fraction<1 or abs(fraction-record['jax_pool_bytes']/record['gpu_total_bytes'])>1e-12:
        raise ValueError('JAX pool does not match the guard')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,default=Path(__file__).resolve().parents[1]/'research/english-gpu-cuda13-linux-x86_64-py312.lock.txt')
    a=p.parse_args()
    if a.output.exists():
        p.error('Preserve earlier backend evidence; choose a new output directory')
    # No JAX import until the actual parent/receipt/device binding is checked.
    parent=os.getppid()
    command=Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
    if not any(Path(x.decode()).name=='guard_english_gpu_job.py' for x in command if x):
        raise ValueError('Run through the GPU admission guard')
    report=Path(os.environ['ACK_ENGLISH_GPU_GUARD_REPORT'])
    deadline=time.monotonic()+3
    while True:
        record=json.loads(report.read_text())
        if record.get('child_pid')==os.getpid() or time.monotonic()>=deadline:
            break
        time.sleep(.05)
    validate_context(record,os.environ,os.getpid(),parent)
    packages={}
    for line in a.lock.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        pin=line.split(' ',1)[0];name,version=pin.split('==')
        if importlib.metadata.version(name)!=version:
            raise ValueError('GPU environment differs from its lock: '+name)
        packages[name]=version
    os.umask(0o077);a.output.mkdir(parents=True)
    (a.output/'INCOMPLETE').write_text('GPU numeric smoke incomplete.\n')
    admission=a.output/'admitted-guard.json'
    admission.write_text(json.dumps(record,indent=2)+'\n')
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update('jax_default_matmul_precision','highest')
    devices=jax.devices()
    if jax.default_backend()!='gpu' or len(devices)!=1 or devices[0].platform!='gpu':
        raise ValueError('Exactly one GPU backend is required; no CPU fallback')
    rng=np.random.default_rng(0)
    x=rng.normal(size=(64,64)).astype(np.float32)
    w=rng.normal(size=(64,8)).astype(np.float32)
    actual=np.asarray(jax.jit(lambda u,v:u@v)(jnp.asarray(x),jnp.asarray(w)))
    expected=x@w
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
    gradient=np.asarray(jax.jit(jax.grad(lambda v:jnp.mean((jnp.asarray(x)@v)**2)))(jnp.asarray(w)))
    grad_expected=(np.float32(2/(64*8))*x.T)@(x@w)
    np.testing.assert_allclose(gradient,grad_expected,rtol=5e-5,atol=5e-5)
    image=rng.normal(size=(1,8,8,1)).astype(np.float32)
    kernel=rng.normal(size=(3,3,1,2)).astype(np.float32)
    conv=np.asarray(jax.jit(lambda u,v:jax.lax.conv_general_dilated(u,v,(1,1),'VALID',
        dimension_numbers=('NHWC','HWIO','NHWC'),precision=jax.lax.Precision.HIGHEST))(jnp.asarray(image),jnp.asarray(kernel)))
    conv_expected=np.stack([np.stack([np.einsum('ijc,ijco->o',image[0,i:i+3,j:j+3],kernel) for j in range(6)]) for i in range(6)])[None,:]
    np.testing.assert_allclose(conv,conv_expected,rtol=2e-5,atol=2e-5)
    result=dict(schema=1,scope=__doc__,guard_receipt_sha256=hashlib.sha256(admission.read_bytes()).hexdigest(),
                guard_receipt_is_preserved_admission_snapshot=True,
                lock_sha256=hashlib.sha256(a.lock.read_bytes()).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                device_kind=devices[0].device_kind,backend=jax.default_backend(),visible_device_count=len(devices),
                packages=packages,matmul_max_absolute_error=float(np.max(np.abs(actual-expected))),
                gradient_max_absolute_error=float(np.max(np.abs(gradient-grad_expected))),
                convolution_max_absolute_error=float(np.max(np.abs(conv-conv_expected))),
                all_passed=True,training_started=False,neural_accuracy_verified=False,deployment_permitted=False)
    (a.output/'backend.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'INCOMPLETE').unlink()
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
