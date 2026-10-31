"""Checkpoint cadence and guarded backend admission for accelerated training."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_due(step,every,last_saved,final=False):
    if type(step) is not int or step<0 or type(every) is not int or every<1:
        raise ValueError('Invalid checkpoint cadence')
    if last_saved is not None and (type(last_saved) is not int or last_saved<0 or last_saved>step):
        raise ValueError('Checkpoint step regressed')
    return step!=last_saved and (final or (step>0 and step%every==0))


class StopRequest:
    """Ask the training loop to checkpoint at its next complete step boundary."""
    def __enter__(self):
        self.requested=False
        self.previous={s:signal.signal(s,self.request) for s in (signal.SIGTERM,signal.SIGINT)}
        return self

    def request(self,*_):
        self.requested=True

    def __exit__(self,*_):
        for s,handler in self.previous.items():
            signal.signal(s,handler)


def backend_proof(path,current_guard,checker,lock,package_version=importlib.metadata.version):
    proof=json.loads(path.read_text())
    if proof.get('all_passed') is not True or proof.get('backend')!='gpu' or proof.get('visible_device_count')!=1:
        raise ValueError('Successful single-GPU backend smoke is required')
    if proof.get('checker_sha256')!=sha(checker) or proof.get('lock_sha256')!=sha(lock):
        raise ValueError('Backend checker or environment lock differs')
    admission=path.parent/'admitted-guard.json'
    if proof.get('guard_receipt_is_preserved_admission_snapshot') is not True or proof.get('guard_receipt_sha256')!=sha(admission):
        raise ValueError('Backend admission snapshot differs')
    old=json.loads(admission.read_text())
    if old.get('device_fingerprint')!=current_guard['device_fingerprint'] or old.get('admitted') is not True:
        raise ValueError('Backend was verified on a different GPU')
    packages=proof['packages']
    expected={line.split(' ',1)[0].split('==')[0]:line.split(' ',1)[0].split('==')[1]
              for line in lock.read_text().splitlines() if line and not line.startswith('#')}
    if packages!=expected or any(package_version(name)!=value for name,value in expected.items()):
        raise ValueError('GPU environment changed since backend verification')
    return proof


def prepare_execution(backend,proof_path=None):
    if backend=='cpu':
        os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='-1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        return dict(backend='cpu',matmul_precision='highest'),None
    if backend!='cuda' or proof_path is None:
        raise ValueError('CUDA requires a verified backend receipt')
    from check_english_gpu_backend import validate_context
    parent=os.getppid()
    command=Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
    if not any(Path(x.decode()).name=='guard_english_gpu_job.py' for x in command if x):
        raise ValueError('GPU training must be launched by its admission guard')
    report=Path(os.environ['ACK_ENGLISH_GPU_GUARD_REPORT'])
    deadline=time.monotonic()+3
    while True:
        guard=json.loads(report.read_text())
        if guard.get('child_pid')==os.getpid() or time.monotonic()>=deadline:
            break
        time.sleep(.05)
    validate_context(guard,os.environ,os.getpid(),parent)
    root=Path(__file__).resolve().parents[1]
    proof=backend_proof(proof_path,guard,root/'tools/check_english_gpu_backend.py',
                        root/'research/english-gpu-cuda13-linux-x86_64-py312.lock.txt')
    # Hardware/receipt identity is run evidence, not an optimizer-compatibility
    # field. Another board needs its own valid smoke before resuming there.
    signature=dict(backend='cuda',matmul_precision='highest',gpu_packages=proof['packages'],
                   backend_checker_sha256=proof['checker_sha256'],environment_lock_sha256=proof['lock_sha256'])
    evidence=dict(guard=guard,backend_receipt_sha256=sha(proof_path),device_fingerprint=guard['device_fingerprint'])
    return signature,evidence
