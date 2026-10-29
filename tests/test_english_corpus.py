"""Corpus safety and split integrity; fake FLAC headers test metadata, not decoding."""
import io
from pathlib import Path
import runpy
import tarfile
import tempfile
import unittest

m = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/prepare_english_corpus.py'))


class EnglishCorpusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def clip(self, speaker='1', chapter='2', utterance='0001', samples=32000, bits=16):
        root = self.root / 'corpus'
        folder = root / speaker / chapter
        folder.mkdir(parents=True, exist_ok=True)
        identity = '-'.join((speaker, chapter, utterance))
        audio = folder / (identity + '.flac')
        packed = (16000 << 44) | ((bits - 1) << 36) | samples
        audio.write_bytes(b'fLaC\x80\x00\x00\x22' + bytes(10) + packed.to_bytes(8, 'big') + bytes(16) + identity.encode())
        transcript = folder / (speaker + '-' + chapter + '.trans.txt')
        with transcript.open('a') as f:
            f.write(identity + ' THE ORIGINAL WORDS\n')
        return root, audio, transcript

    def test_original_text_and_full_audio_preserved(self):
        root, audio, transcript = self.clip()
        before = audio.read_bytes(), transcript.read_bytes()
        rows, _, stats = m['records_for'](root, 'train', 8)
        self.assertEqual(rows[0]['text'], 'THE ORIGINAL WORDS')
        self.assertEqual(rows[0]['language'], 'en')
        self.assertEqual(rows[0]['seconds'], 2)
        self.assertEqual(stats['retained'], 1)
        self.assertEqual(before, (audio.read_bytes(), transcript.read_bytes()))

    def test_long_clip_excluded_whole_and_still_protected_from_leakage(self):
        root, _, _ = self.clip()
        _, audio, _ = self.clip(speaker='3', samples=16000 * 9)
        before = audio.read_bytes()
        rows, groups, stats = m['records_for'](root, 'train', 8)
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats['excluded_over_limit'], 1)
        self.assertIn('3', groups['speakers'])
        self.assertEqual(before, audio.read_bytes())

    def test_wrong_flac_geometry_refused(self):
        _, audio, _ = self.clip(bits=24)
        with self.assertRaises(ValueError):
            m['flac_samples'](audio)

    def test_truncated_flac_refused(self):
        _, audio, _ = self.clip()
        audio.write_bytes(b'fLaC')
        with self.assertRaises(ValueError):
            m['flac_samples'](audio)

    def test_unpaired_audio_or_transcript_refused(self):
        root, audio, _ = self.clip()
        orphan = audio.with_name('1-2-9999.flac')
        orphan.write_bytes(audio.read_bytes())
        with self.assertRaises(ValueError):
            m['records_for'](root, 'train', 8)
        orphan.unlink()
        audio.unlink()
        with self.assertRaises(FileNotFoundError):
            m['records_for'](root, 'train', 8)

    def test_duplicate_utterance_refused(self):
        root, _, transcript = self.clip()
        with transcript.open('a') as f:
            f.write(transcript.read_text())
        with self.assertRaises(ValueError):
            m['records_for'](root, 'train', 8)

    def test_each_kind_of_cross_split_leakage_refused(self):
        for field in ('speakers', 'ids', 'audio_hashes'):
            a = {key: set() for key in ('speakers', 'ids', 'audio_hashes')}
            b = {key: set() for key in a}
            a[field] = b[field] = {'overlap'}
            with self.subTest(field=field), self.assertRaises(ValueError):
                m['assert_disjoint']({'train': a, 'test': b})

    def archive(self, name, link=False):
        path = self.root / 'candidate.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            entry = tarfile.TarInfo(name)
            if link:
                entry.type, entry.linkname = tarfile.SYMTYPE, '/outside'
                archive.addfile(entry)
            else:
                entry.size = 4
                archive.addfile(entry, io.BytesIO(b'data'))
        return path

    def test_archive_escape_and_link_refused(self):
        for index, (name, link) in enumerate((('../outside', False), ('/outside', False), ('LibriSpeech/dev-clean/link', True))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m['extract_split'](self.archive(name, link), 'dev-clean', self.root / str(index))

    def test_repeated_extraction_cannot_overwrite_existing_data(self):
        archive = self.archive('LibriSpeech/dev-clean/1/2/data.txt')
        target = self.root / 'extracted'
        m['extract_split'](archive, 'dev-clean', target)
        with self.assertRaises(FileExistsError):
            m['extract_split'](archive, 'dev-clean', target)
        self.assertEqual((target / '1/2/data.txt').read_bytes(), b'data')


if __name__ == '__main__':
    unittest.main()
