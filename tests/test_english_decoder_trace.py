"""Require host instrumentation to fail closed when engine anchors drift."""
from pathlib import Path
import re
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]
instrument = runpy.run_path(str(ROOT/'tools/prepare_decoder_trace.py'))['instrument']


class DecoderTraceTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT/'engine/src/nd_whistle.c').read_text()

    def test_only_capture_calls_are_added(self):
        generated = instrument(self.source)
        restored, count = re.subn(r'    trace_[a-z_]+\([^;\n]*\);\n', '', generated)
        self.assertEqual(count, 13)
        self.assertEqual(restored, self.source)

    def test_missing_boundary_refused(self):
        with self.assertRaisesRegex(ValueError, 'anchor'):
            instrument(self.source.replace('    /* Engram injection at its sites. */', ''))

    def test_ambiguous_boundary_refused(self):
        anchor = '    /* Engram injection at its sites. */'
        with self.assertRaisesRegex(ValueError, 'anchor'):
            instrument(self.source.replace(anchor, anchor+'\n'+anchor))

    def test_missing_decoder_refused(self):
        with self.assertRaises(ValueError):
            instrument(self.source.replace('static void dec_block(', 'static void changed_block('))

    def test_missing_cross_projection_refused(self):
        with self.assertRaisesRegex(ValueError, 'projection anchor'):
            instrument(self.source.replace('    lin(w, X + X_OUT, o, 1, t2);', ''))


if __name__ == '__main__':
    unittest.main()
