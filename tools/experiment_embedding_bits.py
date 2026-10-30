#!/usr/bin/env python3
"""Research-only CQ4 -> CQ2 tied embedding conversion of the locked English model.

Preserves token IDs, original group norms, codebooks and every other tensor.
This is lossy quantization, not English neural training or a release artifact.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

HEADER = struct.Struct('<48If')
RECORD = struct.Struct('<BBHIIIIQQII')
LOCKED = '8c956a951107b03a1832a3c6d665c88e52cc59fadcf91397077416f22f5dca9b'


def read_model(data):
    h = HEADER.unpack_from(data)
    if h[0] != 0x05E12A84 or h[2] != 28 or h[5:8] != (8193, 0, 512):
        raise ValueError('Expected locked English v4 geometry')
    directory = HEADER.size + h[2] * 4
    previous_end = directory + h[1] * RECORD.size
    records = []
    for index in range(h[1]):
        record = list(RECORD.unpack_from(data, directory + index * RECORD.size))
        offset, size = record[7:9]
        if offset % 64 or offset < previous_end or offset + size > len(data):
            raise ValueError('Tensor bounds or alignment differ')
        records.append((record, data[offset:offset + size]))
        previous_end = offset + size
    if records[-1][0][0] != 4 or previous_end != len(data):
        raise ValueError('Expected final tokenizer without trailing bytes')
    return h, data[HEADER.size:directory], records


def convert(data):
    h, books, tensors = read_model(data)
    record, embedding = tensors[0]
    if record[0:2] != [3, 2] or record[3:5] != [8193, 512] or record[9:] != [128, 4]:
        raise ValueError('Expected CQ4 8193x512 embedding with group128')
    packed_bytes = 8193 * 512 // 2
    if len(embedding) != packed_bytes + 8193 * 4 * 2:
        raise ValueError('Embedding payload length differs')
    codebooks = struct.unpack('<28f', books)
    if not all(math.isfinite(x) for x in codebooks):
        raise ValueError('Nonfinite quantizer codebook')
    two, four = codebooks[:4], codebooks[12:]
    mapping = [min(range(4), key=lambda index: abs(two[index] - value)) for value in four]
    old = embedding[:packed_bytes]
    packed = bytearray(packed_bytes // 2)
    for index in range(0, packed_bytes, 2):
        a, b = old[index:index + 2]
        packed[index // 2] = mapping[a & 15] | (mapping[a >> 4] << 2) | (mapping[b & 15] << 4) | (mapping[b >> 4] << 6)
    norms = embedding[packed_bytes:]
    record[10] = 2
    tensors[0] = record, bytes(packed) + norms
    offset = (HEADER.size + len(books) + len(tensors) * RECORD.size + 63) // 64 * 64
    directory, body = bytearray(), bytearray()
    for index, (record, blob) in enumerate(tensors):
        record[7], record[8] = offset + len(body), len(blob)
        directory += RECORD.pack(*record)
        body += blob
        if index + 1 < len(tensors):
            body += bytes(-len(body) % 64)
    result = HEADER.pack(*h) + books + directory
    result += bytes(offset - len(result))
    result += body
    _, new_books, checked = read_model(result)
    _, old_books, original = read_model(data)
    assert new_books == old_books and HEADER.unpack_from(result) == h
    assert all(a[1] == b[1] for a, b in zip(original[1:], checked[1:]))
    assert checked[0][1][len(packed):] == norms
    return bytes(result), mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error('Choose new artifact and report paths; preserve previous evidence')
    data = args.model.read_bytes()
    if hashlib.sha256(data).hexdigest() != LOCKED:
        parser.error('Input differs from the locked English artifact')
    result, mapping = convert(data)
    with args.output.open('xb') as output:
        output.write(result)
    report = {'scope': 'Research-only lossy tied embedding quantization; not a release',
              'source_sha256': LOCKED, 'candidate_sha256': hashlib.sha256(result).hexdigest(),
              'source_bytes': len(data), 'candidate_bytes': len(result),
              'flash_bytes_saved': len(data) - len(result), 'embedding_bits_before': 4,
              'embedding_bits_after': 2, 'index_mapping': mapping,
              'token_ids_unchanged': True, 'codebooks_unchanged': True,
              'group_norm_bytes_unchanged': True, 'other_tensor_bytes_unchanged': True,
              'recognition_accuracy_verified': False, 'neural_training_run': False,
              'monolingual_neural_model_proven': False, 'deployment_permitted': False}
    with args.report.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print('Research candidate:', len(result), 'bytes;', len(data) - len(result), 'saved; accuracy unverified')


if __name__ == '__main__':
    main()
