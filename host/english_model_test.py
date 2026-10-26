#!/usr/bin/env python3
"""Structural regression for lossless retained rows and stable tokenizer IDs."""
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import whistle_english as en


def records(data):
    h = en.HEADER.unpack_from(data)
    start = en.HEADER.size + 4 * h[2]
    result = []
    for i in range(h[1]):
        r = en.RECORD.unpack_from(data, start + i * en.RECORD.size)
        result.append((r, data[r[7]:r[7] + r[8]]))
    return h, result


def fixture():
    h = [0] * 48 + [10000.]
    h[0], h[1], h[2], h[5], h[7], h[10] = 0x05E12A84, 451, 28, en.OLD, 512, 2
    tokenizer = struct.pack('<5IBBH', en.OLD, 0, 1, 2, 3, 0, 1, 0)
    for i in range(en.OLD):
        text = (f'piece{i}'.encode() if i < en.LANG_BASE else en.LANGUAGES[i-en.LANG_BASE])
        tokenizer += struct.pack('<fBH', -float(i), 0 if i < en.LANG_BASE else 3, len(text)) + text
    packed = bytes(range(256)) * en.OLD
    norms = b''.join(struct.pack('<4H', i, i, i, i) for i in range(en.OLD))
    blobs = [packed + norms] + [b'\x00\x00'] * (h[1]-2) + [tokenizer]
    blobs[67] = struct.pack('<13f', 3246, 1, 8192, 8, 4, 9, 128, 3, 80, 16000, 512, 400, 160)
    offset = (en.HEADER.size + 112 + h[1] * en.RECORD.size + 63) // 64 * 64
    directory, body = bytearray(), bytearray()
    for i, blob in enumerate(blobs):
        r = [1, 1, 0, 1, 0, 0, 0, offset + len(body), len(blob), 0, 0]
        if i == 0: r[0:5] = [3, 2, 0, en.OLD, 512]; r[9:] = [128, 4]
        elif i == 67: r[0:4] = [2, 1, 0, 13]
        elif i == h[1]-1: r[0:4] = [4, 0, 0, 0]
        directory += en.RECORD.pack(*r)
        body += blob
        if i+1 < len(blobs): body += bytes(-len(body) % 64)
    head = en.HEADER.pack(*h) + bytes(112) + directory
    return head + bytes(offset-len(head)) + body


class EnglishModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.source = fixture()

    def test_retained_rows_and_ids_unchanged(self):
        before_h, before = records(self.source)
        out = en.english_bytes(self.source)
        after_h, after = records(out)
        self.assertEqual(en.english_bytes(self.source), out)
        want = list(before_h); want[5] = en.KEEP
        self.assertEqual(tuple(want), after_h)
        self.assertLess(len(out), len(self.source))
        self.assertEqual(after[0][0][3], en.KEEP)
        rb, nb = 256, 8
        for i in range(en.KEEP):
            self.assertEqual(before[0][1][i*rb:(i+1)*rb], after[0][1][i*rb:(i+1)*rb])
            self.assertEqual(before[0][1][en.OLD*rb+i*nb:en.OLD*rb+(i+1)*nb],
                             after[0][1][en.KEEP*rb+i*nb:en.KEEP*rb+(i+1)*nb])
        for i in range(1, len(before)-1): self.assertEqual(before[i][1], after[i][1])
        old_tok, new_tok = before[-1][1], after[-1][1]
        self.assertEqual(struct.unpack_from('<I', new_tok)[0], en.KEEP)
        self.assertEqual(new_tok[4:], old_tok[4:len(new_tok)])
        for marker in en.LANGUAGES[1:]: self.assertNotIn(marker, new_tok)
        self.assertIn(en.LANGUAGES[0], new_tok)
        self.assertTrue(all(r[7] % 64 == 0 for r, _ in after))

    def test_reject_second_prune(self):
        with self.assertRaises(ValueError): en.english_bytes(en.english_bytes(self.source))

    def test_reject_wrong_language_ids(self):
        bad = self.source.replace(b'<|de|>', b'<|xx|>')
        with self.assertRaises(ValueError): en.english_bytes(bad)

    def test_reject_truncation(self):
        for size in (0, 195, 1000, len(self.source)-1):
            with self.assertRaises(ValueError): en.english_bytes(self.source[:size])

    def test_reject_embedding_geometry(self):
        bad = bytearray(self.source)
        struct.pack_into('<I', bad, en.HEADER.size + 112 + 8, en.OLD - 1)
        with self.assertRaises(ValueError): en.english_bytes(bad)


if __name__ == '__main__': unittest.main()
