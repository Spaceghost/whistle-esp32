#!/usr/bin/env python3
"""Remove Whistle's six foreign-language conditioning rows and token markers.

Keep text IDs 0..8191 and English prefix ID 8192 unchanged. Shared text pieces,
encoder/decoder layers and engram tables are preserved: removing arbitrary
foreign-looking text pieces would also damage English names and token hashes.
This is conditioning-row pruning, not an English-only retraining claim.
"""
import argparse
import hashlib
from pathlib import Path
import struct

HEADER = struct.Struct('<48If')
RECORD = struct.Struct('<BBHIIIIQQII')
LANG_BASE = 8192
LANGUAGES = (b'<|en|>', b'<|de|>', b'<|fr|>', b'<|es|>', b'<|it|>', b'<|nl|>', b'<|pl|>')
KEEP = LANG_BASE + 1
OLD = LANG_BASE + len(LANGUAGES)


def read_model(data):
    if len(data) < HEADER.size:
        raise ValueError('truncated header')
    h = list(HEADER.unpack_from(data))
    if h[0] != 0x05E12A84 or h[5] != OLD or h[6] != 0 or h[7] != 512:
        raise ValueError('expected original Whistle v4 tied vocabulary (8199 x 512)')
    if not 2 <= h[10] <= 8 or h[31] > 4 or h[2] != 28:
        raise ValueError('unsupported Whistle geometry')
    directory = HEADER.size + 4 * h[2]
    end = directory + h[1] * RECORD.size
    if end > len(data):
        raise ValueError('truncated directory')
    records = []
    previous_end = end
    for i in range(h[1]):
        r = list(RECORD.unpack_from(data, directory + i * RECORD.size))
        off, size = r[7:9]
        if r[1] > 4 or off % 64 or off < previous_end or off + size > len(data):
            raise ValueError('invalid tensor bounds/order')
        records.append((r, data[off:off + size]))
        previous_end = off + size
    audio_index = 1 + 27 * h[10] + 9 + 2 + 4 * h[31] + 1
    expected_count = audio_index + 1 + 10 * h[10] + 1 + 8 * 43 + 9 + 1 + 6 + 2
    if h[1] != expected_count:
        raise ValueError('unsupported tensor schema')
    r, audio = records[audio_index]
    if r[0] != 2 or r[1] != 1 or r[3] != 13 or len(audio) != 52:
        raise ValueError('invalid audio manifest')
    a = struct.unpack('<13f', audio)
    if a[2] != LANG_BASE or a[3] != 8 or a[8:] != (80., 16000., 512., 400., 160.):
        raise ValueError('unexpected Whistle audio geometry')
    return h, data[HEADER.size:directory], records


def prune_tokenizer(blob):
    if len(blob) < 24 or struct.unpack_from('<I', blob)[0] != OLD:
        raise ValueError('unexpected tokenizer vocabulary')
    off = 24
    kept_end = None
    for i in range(OLD):
        if off + 7 > len(blob):
            raise ValueError('truncated tokenizer record')
        _, kind, size = struct.unpack_from('<fBH', blob, off)
        start = off + 7
        off = start + size
        if off > len(blob):
            raise ValueError('truncated tokenizer surface')
        if i >= LANG_BASE and (kind != 3 or blob[start:off] != LANGUAGES[i - LANG_BASE]):
            raise ValueError('language IDs differ; refusing to remap tokens')
        if i == LANG_BASE:
            kept_end = off
    if off != len(blob):
        raise ValueError('unexpected tokenizer trailing data')
    out = bytearray(blob[:kept_end])
    struct.pack_into('<I', out, 0, KEEP)
    return bytes(out)


def english_bytes(data):
    h, codebook, tensors = read_model(data)
    r, embedding = tensors[0]
    if r[0] != 3 or r[1] != 2 or r[3:5] != [OLD, 512] or r[9:] != [128, 4]:
        raise ValueError('unsupported embedding quantization')
    row_bytes = 512 * 4 // 8
    norm_bytes = 512 // 128 * 2
    if len(embedding) != OLD * (row_bytes + norm_bytes):
        raise ValueError('embedding length mismatch')
    packed = embedding[:KEEP * row_bytes]
    norms = embedding[OLD * row_bytes:OLD * row_bytes + KEEP * norm_bytes]
    r[3] = KEEP
    tensors[0] = r, packed + norms
    r, tokenizer = tensors[-1]
    if r[0] != 4:
        raise ValueError('expected final RAW tokenizer')
    tensors[-1] = r, prune_tokenizer(tokenizer)
    h[5] = KEEP
    offset = (HEADER.size + len(codebook) + len(tensors) * RECORD.size + 63) // 64 * 64
    directory = bytearray()
    body = bytearray()
    for i, (r, blob) in enumerate(tensors):
        r[7], r[8] = offset + len(body), len(blob)
        directory += RECORD.pack(*r)
        body += blob
        if i + 1 < len(tensors):
            body += bytes(-len(body) % 64)
    out = HEADER.pack(*h) + codebook + directory
    out += bytes(offset - len(out))
    return out + body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    original = args.source.read_bytes()
    result = english_bytes(original)
    # Never overwrite a source model or an earlier immutable artifact.
    with args.output.open('xb') as out:
        out.write(result)
    print(f'Removed 6 language rows/markers: {len(original)} -> {len(result)} bytes '
          f'({len(original) - len(result)} saved); SHA256 {hashlib.sha256(result).hexdigest()}')


if __name__ == '__main__':
    main()
