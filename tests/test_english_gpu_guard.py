"""Fake GPU telemetry with real process isolation; never initializes CUDA."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import guard_english_gpu_job as g


def gpu(**changes):
    value=dict(index=0,uuid='GPU-test',total_bytes=8*g.GIB,free_bytes=7*g.GIB,utilization=0,processes=[])
    value.update(changes)
    return value


class AdmissionTests(unittest.TestCase):
    def test_idle_without_compute_and_enough_memory_admits(self):
        self.assertIsNone(g.admission_reason(gpu(),8*g.GIB,5*g.GIB,5*g.GIB,10))

    def test_other_compute_is_not_idle_even_at_zero_utilization(self):
        self.assertEqual(g.admission_reason(gpu(processes=[{'pid':42,'bytes':1}]),8*g.GIB,5*g.GIB,5*g.GIB,10),'other_compute_process_present')

    def test_busy_display_and_low_memory_refuse(self):
        for device,ram,expected in ((gpu(utilization=20),8*g.GIB,'gpu_utilization_above_admission_limit'),
            (gpu(free_bytes=4*g.GIB),8*g.GIB,'gpu_memory_headroom'),(gpu(),4*g.GIB,'host_memory_headroom')):
            with self.subTest(expected=expected):
                self.assertEqual(g.admission_reason(device,ram,5*g.GIB,5*g.GIB,10),expected)

    def test_runtime_own_utilization_is_allowed(self):
        self.assertIsNone(g.running_reason(gpu(utilization=100,processes=[{'pid':9,'bytes':g.GIB}]),{9},8*g.GIB,g.GIB,1,3*g.GIB,2*g.GIB,3*g.GIB,g.GIB,30))

    def test_withdrawal_thresholds(self):
        cases=[(gpu(processes=[{'pid':10,'bytes':1}]),g.GIB,8*g.GIB,1,'other_compute_process_arrived'),
               (gpu(processes=[{'pid':9,'bytes':4*g.GIB}]),g.GIB,8*g.GIB,1,'owned_gpu_memory_limit'),
               (gpu(free_bytes=g.GIB),g.GIB,8*g.GIB,1,'gpu_free_memory_floor'),
               (gpu(),4*g.GIB,8*g.GIB,1,'owned_process_group_rss_limit'),
               (gpu(),g.GIB,g.GIB//2,1,'host_available_memory_floor'),
               (gpu(),g.GIB,8*g.GIB,31,'job_time_limit')]
        for device,rss,free,elapsed,expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(g.running_reason(device,{9},free,rss,elapsed,3*g.GIB,2*g.GIB,3*g.GIB,g.GIB,30),expected)

    def test_unavailable_gpu_telemetry_fails_closed(self):
        with patch.object(g.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'0, GPU-test, 8192, 7168, N/A\n','')):
            with self.assertRaises(ValueError):
                g.snapshot(0)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def args(self,command):
        return ['guard','--device','0','--report',str(self.root/'job.json'),'--lock-dir',str(self.root/'locks'),
                '--cache-dir',str(self.root/'cache'),
                '--idle-seconds','0','--wait-seconds','0','--max-seconds','10','--',*command]

    def test_child_receives_only_selected_gpu_and_bounded_pool(self):
        code="import os; assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-test'; assert os.environ['JAX_PLATFORMS']=='cuda'; assert float(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])==.25; assert 'JAX_SKIP_CUDA_CONSTRAINTS_CHECK' not in os.environ"
        with patch.object(sys,'argv',self.args([sys.executable,'-c',code])),patch.object(g,'snapshot',return_value=gpu()),patch.object(g,'available_memory',return_value=8*g.GIB):
            self.assertEqual(g.main(),0)
        self.assertEqual(json.loads((self.root/'job.json').read_text())['state'],'complete')

    def test_foreign_arrival_stops_owned_child_and_preserves_other_process(self):
        other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            sequence=[gpu(),gpu(),gpu(processes=[{'pid':other.pid,'bytes':g.GIB}])]
            with patch.object(sys,'argv',self.args([sys.executable,'-c','import time; time.sleep(30)'])),patch.object(g,'snapshot',side_effect=sequence),patch.object(g,'available_memory',return_value=8*g.GIB):
                self.assertEqual(g.main(),1)
            result=json.loads((self.root/'job.json').read_text())
            self.assertEqual(result['reason'],'other_compute_process_arrived')
            self.assertIsNone(other.poll())
            self.assertFalse(g.group_pids(result['child_pid']))
        finally:
            other.terminate();other.wait(timeout=5)

    def test_busy_gpu_never_starts_child(self):
        marker=self.root/'started'
        with patch.object(sys,'argv',self.args([sys.executable,'-c',f"open({str(marker)!r},'w').close()"])),patch.object(g,'snapshot',return_value=gpu(utilization=50)),patch.object(g,'available_memory',return_value=8*g.GIB):
            self.assertEqual(g.main(),75)
        self.assertFalse(marker.exists())
        self.assertFalse(json.loads((self.root/'job.json').read_text())['admitted'])

    def test_existing_receipt_is_preserved(self):
        (self.root/'job.json').write_text('unique old receipt')
        with patch.object(sys,'argv',self.args([sys.executable,'-c','pass'])):
            with self.assertRaises(SystemExit):
                g.main()
        self.assertEqual((self.root/'job.json').read_text(),'unique old receipt')

    def test_cooperative_gpu_lock_prevents_second_owner(self):
        (self.root/'locks').mkdir()
        fingerprint=hashlib.sha256(b'GPU-test').hexdigest()[:16]
        with (self.root/'locks'/(fingerprint+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(sys,'argv',self.args([sys.executable,'-c','raise SystemExit(9)'])),patch.object(g,'snapshot',return_value=gpu()):
                self.assertEqual(g.main(),75)
        self.assertEqual(json.loads((self.root/'job.json').read_text())['reason'],'cooperating_gpu_guard_already_active')

    def test_busy_sample_resets_required_idle_window(self):
        calls={'snapshot':0,'launched_at':None}
        def sample(_):
            calls['snapshot']+=1
            return gpu(utilization=50 if calls['snapshot']==3 else 0)
        real_popen=subprocess.Popen
        def launch(*args,**kwargs):
            calls['launched_at']=calls['snapshot']
            return real_popen(*args,**kwargs)
        args=self.args([sys.executable,'-c','pass'])
        args[args.index('--idle-seconds')+1]='2'
        args[args.index('--wait-seconds')+1]='10'
        args[args.index('--max-seconds')+1]='1000'
        tick=iter(range(10000));real_sleep=time.sleep
        with patch.object(sys,'argv',args),patch.object(g,'snapshot',side_effect=sample),patch.object(g,'available_memory',return_value=8*g.GIB),patch.object(g.time,'monotonic',side_effect=lambda:next(tick)),patch.object(g.time,'sleep',side_effect=lambda _:real_sleep(.001)),patch.object(g.subprocess,'Popen',side_effect=launch):
            self.assertEqual(g.main(),0)
        self.assertEqual(calls['launched_at'],6)

    def test_runtime_telemetry_failure_reaps_owned_group(self):
        args=self.args([sys.executable,'-c','import time; time.sleep(30)'])
        with patch.object(sys,'argv',args),patch.object(g,'snapshot',side_effect=[gpu(),gpu(),subprocess.TimeoutExpired('nvidia-smi',1)]),patch.object(g,'available_memory',return_value=8*g.GIB):
            with self.assertRaises(subprocess.TimeoutExpired):
                g.main()
        result=json.loads((self.root/'job.json').read_text())
        self.assertEqual(result['reason'],'telemetry_or_guard_error')
        self.assertFalse(g.group_pids(result['child_pid']))


if __name__=='__main__':
    unittest.main()
