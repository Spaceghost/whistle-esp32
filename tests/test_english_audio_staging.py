"""Downloaded archives must satisfy both recorded hashes and exact size."""
import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest

verify=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/stage_english_audio.py'))['verify']


class AudioStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'archive.part'
        self.data=b'original archive bytes'
        self.path.write_bytes(self.data)
        self.record=dict(file='archive.tar.gz',bytes=len(self.data),
                         upstream_md5=hashlib.md5(self.data).hexdigest(),sha256=hashlib.sha256(self.data).hexdigest())

    def test_exact_source_accepted(self):
        verify(self.path,self.record)

    def test_truncated_source_refused_and_retained(self):
        self.path.write_bytes(self.data[:-1])
        with self.assertRaisesRegex(ValueError,'size'):
            verify(self.path,self.record)
        self.assertTrue(self.path.exists())

    def test_sha_mismatch_refused_and_retained(self):
        self.record['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'checksum'):
            verify(self.path,self.record)
        self.assertEqual(self.path.read_bytes(),self.data)

    def test_md5_mismatch_refused_and_retained(self):
        self.record['upstream_md5']='0'*32
        with self.assertRaisesRegex(ValueError,'checksum'):
            verify(self.path,self.record)


if __name__=='__main__':
    unittest.main()
