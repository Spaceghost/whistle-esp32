"""Backend initialization requires the admitted parent's exact context."""
import hashlib
from pathlib import Path
import runpy
import unittest

validate=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/check_english_gpu_backend.py'))['validate_context']


class BackendContextTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(state='running',admitted=True,guard_pid=10,child_pid=99,
                         device_fingerprint=hashlib.sha256(b'GPU-test').hexdigest()[:16],
                         jax_pool_bytes=2*1024**3,gpu_total_bytes=8*1024**3)
        self.env=dict(ACK_ENGLISH_GPU_GUARD_PID='10',CUDA_VISIBLE_DEVICES='GPU-test',
                      JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.25')

    def test_exact_guard_context_accepted(self):
        validate(self.record,self.env,99,10)

    def test_unadmitted_or_wrong_process_refused(self):
        for key,value in (('admitted',False),('state','waiting'),('child_pid',98),('guard_pid',11)):
            with self.subTest(key=key):
                record=dict(self.record,**{key:value})
                with self.assertRaises(ValueError):
                    validate(record,self.env,99,10)

    def test_device_and_backend_override_refused(self):
        for key,value in (('CUDA_VISIBLE_DEVICES','GPU-other'),('CUDA_VISIBLE_DEVICES','GPU-test,GPU-other'),
                          ('JAX_PLATFORMS','cpu'),('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate(self.record,dict(self.env,**{key:value}),99,10)

    def test_allocation_override_refused(self):
        for value in ('.75','nan','0','1'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate(self.record,dict(self.env,XLA_PYTHON_CLIENT_MEM_FRACTION=value),99,10)


if __name__=='__main__':
    unittest.main()
