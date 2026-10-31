"""Durable cadence, cooperative stop and prior GPU verification contracts."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import english_training_control as c


class CadenceTests(unittest.TestCase):
    def test_resume_uses_global_step_boundary(self):
        self.assertFalse(c.checkpoint_due(99,100,None))
        self.assertTrue(c.checkpoint_due(100,100,None))

    def test_final_snapshot_is_saved_once(self):
        self.assertTrue(c.checkpoint_due(103,100,100,final=True))
        self.assertFalse(c.checkpoint_due(103,100,103,final=True))

    def test_initialized_zero_is_only_saved_on_final(self):
        self.assertFalse(c.checkpoint_due(0,10,None))
        self.assertTrue(c.checkpoint_due(0,10,None,final=True))

    def test_invalid_or_regressing_steps_refused(self):
        for step,every,last in ((1,0,None),(True,10,None),(-1,10,None),(1,10,2),(1,10,-1)):
            with self.subTest(step=step,every=every,last=last):
                with self.assertRaises(ValueError):c.checkpoint_due(step,every,last)

    def test_sigterm_sets_request_and_restores_handlers(self):
        previous=signal.getsignal(signal.SIGTERM)
        with c.StopRequest() as stop:
            self.assertFalse(stop.requested)
            os.kill(os.getpid(),signal.SIGTERM)
            self.assertTrue(stop.requested)
        self.assertEqual(signal.getsignal(signal.SIGTERM),previous)

    def test_cpu_default_cannot_initialize_cuda(self):
        with patch.dict(os.environ,{},clear=True):
            signature,evidence=c.prepare_execution('cpu')
            self.assertEqual(signature['backend'],'cpu')
            self.assertIsNone(evidence)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'-1')
            self.assertEqual(os.environ['JAX_PLATFORMS'],'cpu')

    def test_cuda_without_prior_backend_receipt_refused(self):
        with self.assertRaisesRegex(ValueError,'verified backend'):
            c.prepare_execution('cuda')


class BackendProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.checker=self.root/'checker.py';self.checker.write_text('verified checker')
        self.lock=self.root/'gpu.lock';self.lock.write_text('jax==1 --hash=sha256:known\n')
        self.admission=self.root/'admitted-guard.json'
        self.admission.write_text(json.dumps({'device_fingerprint':'same-device','admitted':True}))
        self.path=self.root/'backend.json'
        self.data=dict(all_passed=True,backend='gpu',visible_device_count=1,checker_sha256=c.sha(self.checker),
                       lock_sha256=c.sha(self.lock),guard_receipt_is_preserved_admission_snapshot=True,
                       guard_receipt_sha256=c.sha(self.admission),packages={'jax':'1'})

    def verify(self,version=lambda _: '1'):
        self.path.write_text(json.dumps(self.data))
        return c.backend_proof(self.path,{'device_fingerprint':'same-device'},self.checker,self.lock,version)

    def test_matching_gpu_and_environment_accepted(self):
        self.assertTrue(self.verify()['all_passed'])

    def test_failed_smoke_and_cpu_fallback_refused(self):
        for key,value in (('all_passed',False),('backend','cpu'),('visible_device_count',2)):
            with self.subTest(key=key):
                original=self.data[key];self.data[key]=value
                with self.assertRaises(ValueError):self.verify()
                self.data[key]=original

    def test_changed_software_or_admission_refused(self):
        for path in (self.checker,self.lock,self.admission):
            with self.subTest(path=path.name):
                original=path.read_text();path.write_text('changed')
                with self.assertRaises((ValueError,KeyError)):self.verify()
                path.write_text(original)

    def test_environment_drift_refused(self):
        with self.assertRaisesRegex(ValueError,'environment changed'):
            self.verify(lambda _: 'different-version')

    def test_other_physical_gpu_refused(self):
        self.admission.write_text(json.dumps({'device_fingerprint':'other-device','admitted':True}))
        self.data['guard_receipt_sha256']=c.sha(self.admission)
        with self.assertRaisesRegex(ValueError,'different GPU'):self.verify()


if __name__=='__main__':
    unittest.main()
