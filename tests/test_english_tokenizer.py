"""Prevent held-out or changed corpus data from fitting English tokenizers."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest

m = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/train_english_tokenizer.py'))


class EnglishTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = {}
        for i, split in enumerate(('train', 'validation', 'test')):
            self.rows[split] = [{'id': str(i), 'speaker': str(i), 'audio_sha256': str(i),
                                 'text': 'ORIGINAL ENGLISH TEXT', 'language': 'en', 'split': split}]
        self.write()

    def write(self):
        manifests = {}
        for split, rows in self.rows.items():
            path = self.root / (split + '.jsonl')
            path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            manifests[split] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (self.root / 'corpus.json').write_text(json.dumps({
            'source_declared_language': 'en', 'speaker_and_id_and_encoded_audio_disjoint': True,
            'manifests': manifests}))

    def test_verified_splits_keep_their_roles(self):
        splits, _ = m['load_splits'](self.root)
        self.assertEqual(splits['train'], self.rows['train'])
        self.assertNotEqual(splits['train'], splits['test'])

    def test_changed_manifest_refused_before_training(self):
        with (self.root / 'train.jsonl').open('a') as f:
            f.write('{}\n')
        with self.assertRaises(ValueError):
            m['load_splits'](self.root)

    def test_foreign_or_mixed_split_metadata_refused(self):
        for key, value in (('language', 'de'), ('split', 'test')):
            self.rows['train'][0][key] = value
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                m['load_splits'](self.root)
            self.rows['train'][0][key] = {'language': 'en', 'split': 'train'}[key]

    def test_rechecked_leakage_ignores_claimed_success_flag(self):
        for key in ('speaker', 'id', 'audio_sha256'):
            self.rows['test'][0][key] = self.rows['train'][0][key]
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                m['load_splits'](self.root)
            self.rows['test'][0][key] = '2'

    def test_incomplete_corpus_refused(self):
        (self.root / 'INCOMPLETE').write_text('unfinished')
        with self.assertRaises(ValueError):
            m['load_splits'](self.root)

    def test_casing_views_preserve_source(self):
        self.assertEqual(m['text_views']('THE BOOK'), ['THE BOOK', 'the book', 'The book'])
        self.assertEqual(m['text_views']('a'), ['a', 'A'])


if __name__ == '__main__':
    unittest.main()
