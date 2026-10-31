"""A full-data audit must not mistake sparse staging for a complete corpus."""
import hashlib
import io
from pathlib import Path
import runpy
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
audit=runpy.run_path(str(ROOT/'tools/audit_english_training.py'))['audit_audio']
mapper=runpy.run_path(str(ROOT/'tools/train_english_student.py'))['audio_path']


class TrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache=Path(self.temp.name)
        self.source=Path('/remote/audio')
        self.good=b'unchanged source clip'

    def row(self,name):
        return dict(id=name,audio=str(self.source/(name+'.flac')),audio_sha256=hashlib.sha256(self.good).hexdigest(),samples=16000)

    def test_sparse_cache_stays_incomplete_and_lists_missing(self):
        (self.cache/'one.flac').write_bytes(self.good)
        missing,corrupt=io.StringIO(),io.StringIO()
        stats=audit({'train':[self.row('one'),self.row('two')]},mapper,self.source,self.cache,missing,corrupt)['train']
        self.assertFalse(stats['complete'])
        self.assertEqual((stats['hash_verified'],stats['missing']),(1,1))
        self.assertIn('two.flac',missing.getvalue())
        self.assertEqual(corrupt.getvalue(),'')

    def test_existing_changed_audio_does_not_count_as_verified(self):
        (self.cache/'one.flac').write_bytes(b'changed')
        missing,corrupt=io.StringIO(),io.StringIO()
        stats=audit({'train':[self.row('one')]},mapper,self.source,self.cache,missing,corrupt)['train']
        self.assertEqual(stats['hash_mismatch'],1)
        self.assertEqual(stats['samples_verified'],0)
        self.assertFalse(stats['complete'])
        self.assertIn('one.flac',corrupt.getvalue())

    def test_complete_cache_and_empty_split_are_distinct(self):
        (self.cache/'one.flac').write_bytes(self.good)
        stats=audit({'train':[self.row('one')],'test':[]},mapper,self.source,self.cache,io.StringIO(),io.StringIO())
        self.assertTrue(stats['train']['complete'])
        self.assertFalse(stats['test']['complete'])

    def test_symlink_escape_is_refused(self):
        (self.cache/'one.flac').symlink_to(self.cache.parent/'outside')
        with self.assertRaises(ValueError):
            audit({'train':[self.row('one')]},mapper,self.source,self.cache,io.StringIO(),io.StringIO())


if __name__=='__main__':
    unittest.main()
