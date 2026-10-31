"""Real small-process checks: terminal receipts and isolation of termination."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

GUARD = Path(__file__).parents[1] / 'tools/guard_english_job.py'


class JobGuardTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def command(self, report, seconds, code):
        # Tiny stdlib-only children, no ML allocation. Production ML jobs use
        # the default 1 GiB floor and 2 GiB launch reserve.
        return [sys.executable, str(GUARD), '--report', str(report), '--max-rss-gib', '.25',
                '--host-floor-gib', '0', '--reserve-gib', '0', '--wait-seconds', '0',
                '--admission-samples', '1', '--max-seconds', str(seconds), '--', sys.executable, '-c', code]

    def test_completed_child_and_existing_receipt_are_preserved(self):
        report = self.root / 'complete.json'
        command = self.command(report, 5, "print('small guard smoke')")
        subprocess.run(command, check=True, capture_output=True, timeout=10)
        result = json.loads(report.read_text())
        self.assertEqual((result['state'], result['exit_code']), ('complete', 0))
        self.assertTrue(result['admitted'])
        before = report.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)
        self.assertEqual(report.read_bytes(), before)

    def test_timeout_stops_owned_group_and_leaves_another_process_running(self):
        marker = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(10)'], start_new_session=True)
        def cleanup():
            if marker.poll() is None:marker.terminate()
            marker.wait(timeout=5)
        self.addCleanup(cleanup)
        child_pid = self.root / 'grandchild.pid'
        code = ('import subprocess,sys,time;from pathlib import Path;'
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']);"
                'Path(' + repr(str(child_pid)) + ').write_text(str(p.pid));time.sleep(10)')
        report = self.root / 'timeout.json'
        run = subprocess.run(self.command(report, .35, code), capture_output=True, timeout=10)
        self.assertNotEqual(run.returncode, 0)
        result = json.loads(report.read_text())
        self.assertEqual(result['reason'], 'job_time_limit')
        self.assertIsNone(marker.poll())
        status = Path('/proc') / child_pid.read_text() / 'stat'
        if status.exists():
            text = status.read_text()
            self.assertEqual(text[text.rfind(')') + 2:].split()[0], 'Z')


if __name__ == '__main__':
    unittest.main()
