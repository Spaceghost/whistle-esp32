"""Student data/checkpoint admission without requiring JAX or decoding bulk audio."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest

m = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/train_english_student.py'))


class Tokenizer:
    def encode(self, text):
        return list(range(4, 4 + len(text.split())))

    def get_piece_size(self):
        return 2048

    def unk_id(self):
        return 3


class EnglishStudentGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_metadata_selection_never_decodes_or_truncates_audio(self):
        rows = [{'id': 'short', 'samples': 16000, 'text': 'THE WORDS'},
                {'id': 'long', 'samples': 480001, 'text': 'THE ORIGINAL LONG RECORDING'},
                {'id': 'tokens', 'samples': 16000, 'text': ' '.join(['WORD'] * 40)}]
        selected, stats = m['eligible_rows'](rows, Tokenizer(), 480000, 32)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['token_ids'], [2, 2048, 4, 5, 1])
        self.assertEqual(stats['excluded_duration'], 1)
        self.assertEqual(stats['excluded_tokens'], 1)
        self.assertNotIn('token_ids', rows[0])

    def test_no_eligible_audio_fails_before_model_initialization(self):
        with self.assertRaises(ValueError):
            m['eligible_rows']([{'samples': 480001, 'text': 'ORIGINAL'}], Tokenizer(), 480000, 32)

    def test_sampler_resume_uses_global_step_across_epoch_boundary(self):
        full = [m['sample_indices'](5, step, 2, 7) for step in range(6)]
        resumed = [m['sample_indices'](5, step, 2, 7) for step in range(3, 6)]
        self.assertEqual(full[3:], resumed)
        self.assertEqual(sorted(sum(full[:3], [])[:5]), list(range(5)))

    def test_audio_mapping_cannot_escape_cache(self):
        source, local = self.root / 'original', self.root / 'cache'
        local.mkdir()
        row = {'audio': str(source / 'speaker/clip.flac')}
        self.assertEqual(m['audio_path'](row, source, local), local.resolve() / 'speaker/clip.flac')
        with self.assertRaises(ValueError):
            m['audio_path']({'audio': str(self.root / 'outside.flac')}, source, local)
        (local / 'link').symlink_to(self.root / 'outside', target_is_directory=True)
        with self.assertRaises(ValueError):
            m['audio_path']({'audio': str(source / 'link/clip.flac')}, source, local)

    def test_origin_namespace_is_not_resolved_on_the_destination_host(self):
        source, local = self.root / 'remote-srv', self.root / 'cache'
        source.symlink_to(self.root / 'different-local-srv', target_is_directory=True)
        local.mkdir()
        row = {'audio': str(source / 'speaker/clip.flac')}
        self.assertEqual(m['audio_path'](row, source, local), local.resolve() / 'speaker/clip.flac')

    def checkpoint(self):
        path = self.root / 'checkpoint'
        path.mkdir()
        data = path / 'model.safetensors'
        data.write_bytes(b'opaque checkpoint payload')
        (path / 'READY').write_text('complete')
        (path / 'manifest.json').write_text(json.dumps({'files': {'model': {
            'file': data.name, 'bytes': data.stat().st_size,
            'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}}}))
        return path, data

    def test_changed_checkpoint_is_refused_before_deserialization(self):
        root, data = self.checkpoint()
        m['checkpoint_manifest'](root)
        data.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            m['checkpoint_manifest'](root)

    def test_incomplete_checkpoint_is_never_resumed(self):
        root, _ = self.checkpoint()
        (root / 'INCOMPLETE').write_text('unfinished')
        with self.assertRaises(ValueError):
            m['checkpoint_manifest'](root)


if __name__ == '__main__':
    unittest.main()
