"""Synthetic engine/audio controls; these tests make no ASR accuracy claim."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import eval_english_native as scoring


class HeldOutNativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root/'corpus'; self.corpus.mkdir()
        self.cache = self.root/'audio'; self.cache.mkdir()
        self.rows = {}
        for index, split in enumerate(('train', 'validation', 'test'), 1):
            rows = []
            for number in range(2 if split=='validation' else 1):
                identity = f'{index}-1-{number:04}'
                path = self.cache/split/(identity+'.flac'); path.parent.mkdir(exist_ok=True)
                pcm = np.full(160, index*100+number, dtype=np.int16)
                sf.write(path, pcm, 16000, format='FLAC', subtype='PCM_16')
                rows.append(dict(id=identity, speaker=str(index), language='en', split=split,
                    text='ONE TWO' if number==0 else 'ONE TWO THREE', samples=160,
                    audio='/srv/heldout/'+split+'/'+identity+'.flac', audio_sha256=scoring.sha(path)))
            self.rows[split] = rows
        self.write_manifests()
        self.model = self.root/'model.cact'; self.model.write_bytes(b'synthetic model, not a recognizer')
        self.engine = self.root/'engine'
        self.set_engine("print('one two')")

    def set_engine(self, body):
        self.engine.write_text('#!/usr/bin/env python3\nimport sys,time\n'
            "assert sys.argv[3:]==['--lang','en','--depth','2']\n"+body+'\n')
        self.engine.chmod(0o700)

    def write_manifests(self):
        manifests = {}
        for split, rows in self.rows.items():
            path = self.corpus/(split+'.jsonl')
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            manifests[split] = dict(file=path.name, sha256=scoring.sha(path))
        (self.corpus/'corpus.json').write_text(json.dumps(dict(source_declared_language='en',
            speaker_and_id_and_encoded_audio_disjoint=True, source='Synthetic test controls', manifests=manifests)))

    def args(self, name='evaluation', **changes):
        values = dict(corpus=self.corpus, model=self.model, engine=self.engine,
            audio_source_root=Path('/srv/heldout'), audio_local_root=self.cache,
            output=self.root/name, split='validation', limit=0, seed=0, depth=2, timeout=5)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_normalization_and_edit_distance(self):
        self.assertEqual(scoring.words('“Don’t” CALL Levi—please!'), ["don't", 'call', 'levi', 'please'])
        self.assertEqual(scoring.words('one_two'), ['one', 'two'])
        self.assertEqual(scoring.word_errors(['one', 'two'], []), 2)
        self.assertEqual(scoring.word_errors(['one', 'two'], ['one', 'three', 'four']), 2)

    def test_full_split_micro_wer_and_frozen_evidence(self):
        args = self.args(); report = scoring.evaluate(args)
        self.assertTrue(report['complete'] and report['full_split'])
        self.assertEqual((report['clips'], report['reference_words'], report['word_errors']), (2, 5, 1))
        self.assertEqual(report['wer'], 0.2)
        self.assertFalse(report['recognition_accuracy_accepted'] or report['deployment_permitted'])
        self.assertEqual(scoring.read_evaluation(args.output), report)
        # Sources may change later; the actual inputs to this run remain frozen.
        self.model.write_bytes(b'changed source')
        self.set_engine("print('changed source')")
        self.assertEqual(scoring.read_evaluation(args.output), report)
        with (args.output/'utterances.jsonl').open('a') as file: file.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            scoring.read_evaluation(args.output)

    def test_subset_selection_is_reproducible_and_not_a_full_split(self):
        first = scoring.evaluate(self.args('first', limit=1))
        second = scoring.evaluate(self.args('second', limit=1))
        self.assertFalse(first['full_split'])
        self.assertEqual(first['selected_ids_sha256'], second['selected_ids_sha256'])
        self.assertEqual(first['clips'], 1)

    def test_failed_engine_output_is_not_scored_as_a_transcript(self):
        self.set_engine("print('one two');sys.stderr.write('decode failed');sys.exit(7)")
        args = self.args()
        with self.assertRaisesRegex(ValueError, 'exit 7'):
            scoring.evaluate(args)
        self.assertTrue((args.output/'INCOMPLETE').exists())
        self.assertFalse((args.output/'evaluation.json').exists())
        self.assertEqual(json.loads((args.output/'failure.json').read_text())['completed'], 0)

    def test_timeout_is_not_scored_and_leaves_incomplete(self):
        self.set_engine('time.sleep(1)')
        args = self.args(timeout=0.02)
        with self.assertRaises(subprocess.TimeoutExpired): scoring.evaluate(args)
        self.assertTrue((args.output/'INCOMPLETE').exists())
        self.assertFalse((args.output/'evaluation.json').exists())

    def test_empty_hypothesis_counts_every_reference_word(self):
        self.set_engine("print('')")
        report = scoring.evaluate(self.args())
        self.assertEqual(report['wer'], 1)
        self.assertEqual(report['word_errors'], report['reference_words'])

    def test_changed_audio_and_symlink_escape_are_refused(self):
        row = self.rows['validation'][0]
        path = scoring.audio_path(row, Path('/srv/heldout'), self.cache)
        original = path.read_bytes()
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'audio hash differs'):
            scoring.make_wav(row, Path('/srv/heldout'), self.cache, self.root/'clip.wav')
        path.unlink(); outside = self.root/'outside.flac'; outside.write_bytes(original); path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            scoring.make_wav(row, Path('/srv/heldout'), self.cache, self.root/'clip.wav')

    def test_pcm_conversion_preserves_every_sample(self):
        row = self.rows['validation'][0]; target = self.root/'clip.wav'
        scoring.make_wav(row, Path('/srv/heldout'), self.cache, target)
        with wave.open(str(target), 'rb') as file:
            self.assertEqual((file.getnframes(), file.getframerate(), file.getnchannels()), (160, 16000, 1))
            np.testing.assert_array_equal(np.frombuffer(file.readframes(160), dtype='<i2'), np.full(160, 200))

    def test_train_split_and_leaking_receipt_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'held-out'):
            scoring.evaluate(self.args(split='train'))
        self.rows['validation'][0]['speaker']='1'; self.write_manifests()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            scoring.evaluate(self.args())


if __name__ == '__main__':
    unittest.main()
