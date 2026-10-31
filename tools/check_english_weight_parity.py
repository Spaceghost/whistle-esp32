#!/usr/bin/env python3
"""Compare every packed CQ matrix with its actual full-checkpoint QAT values.

CPU only. Use an external resource guard. No speech or deployment acceptance.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'model', 'needle-source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new evidence path')
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    helper = runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    helper['verify_source'](a.needle_source)
    manifest = helper['checkpoint_manifest'](a.checkpoint)
    receipt = json.loads((a.model.parent / 'export.json').read_text())
    if (a.model.parent / 'INCOMPLETE').exists() or sha(a.model) != receipt['sha256'] or sha(a.checkpoint / 'manifest.json') != receipt['checkpoint_manifest_sha256']:
        raise ValueError('Changed or mismatched checkpoint/export')
    signature = manifest['metadata']['signature']
    if signature != receipt['training_signature']:
        raise ValueError('Training signature differs')
    import english_precision
    if sha(Path(english_precision.__file__)) != signature['precision_module_sha256']:
        raise ValueError('Changed precision implementation')
    whistle, quantize, checkpoints = helper['source_modules'](a.needle_source)
    export = importlib.import_module('_ack_needle.model.export')
    import jax
    import numpy as np
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU only')
    saved = checkpoints.read_checkpoint(a.checkpoint / 'model.safetensors')
    cfg = whistle.WhistleConfig.from_saved(saved['config'])
    deployed = english_precision.qat_params(saved['params'], quantize, signature['qat_bits'])
    # Reuse the pinned source's geometry mapping, replacing packing with raw
    # FP32 values so that QAT weights are not quantized a second time.
    original_q = export._q
    def raw_q(name, matrix, bits, group):
        value = np.asarray(matrix, np.float32)
        return export._Tensor(name, export.FP32, value.shape, value.tobytes())
    export._q = raw_q
    try:
        expected = export._tensors(deployed, cfg.needle(), signature['qat_bits'], 128, audio=cfg)
    finally:
        export._q = original_q
    raw = a.model.read_bytes()
    header = struct.unpack_from(export._HDR_FMT, raw)
    if header[0] != 0x31455741 or header[1] != len(expected) + 1:
        raise ValueError('Wrong research container geometry')
    results = []
    offset = struct.calcsize(export._HDR_FMT) + header[2] * 4
    for index, tensor in enumerate(expected):
        rec = struct.unpack_from(export._REC_FMT, raw, offset + index * export.REC_SIZE)
        if rec[0] != export.CQ:
            continue
        shape = tuple(rec[3:3 + rec[1]])
        if shape != tensor.shape or tensor.dtype != export.FP32 or len(shape) != 2:
            raise ValueError(('Wrong CQ mapping', tensor.name))
        group, bits = rec[9:11]
        rows, width = shape
        padded = (width + group - 1) // group * group
        packed_bytes = rows * export._packed_row_bytes(padded, bits, group)
        blob = raw[rec[7]:rec[7] + rec[8]]
        packed = np.frombuffer(blob[:packed_bytes], np.uint8).reshape(rows, -1)
        norms = np.frombuffer(blob[packed_bytes:], np.float16).reshape(rows, padded // group)
        actual = export._cq_unpack(packed, norms, rows, width, bits, group)
        target = np.frombuffer(tensor.blob, np.float32).reshape(shape)
        difference = np.abs(actual - target)
        within = np.isclose(actual, target, atol=2e-6, rtol=2e-5)
        record = dict(name=tensor.name, shape=shape, bits=bits,
                      maximum_absolute_error=float(difference.max()),
                      values_outside_tolerance=int(np.count_nonzero(~within)),
                      within_tolerance=bool(within.all()))
        results.append(record)
        print(json.dumps(record), flush=True)
    report = dict(schema=1, scope='Full-checkpoint CQ packing/QAT comparison only',
                  model_sha256=sha(a.model), checkpoint_manifest_sha256=sha(a.checkpoint / 'manifest.json'),
                  checker_sha256=sha(Path(__file__)), absolute_tolerance=2e-6, relative_tolerance=2e-5,
                  matrices=results, all_passed=bool(results and all(x['within_tolerance'] for x in results)),
                  deployment_permitted=False)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_passed']:
        raise ValueError('Packed weights differ from actual QAT weights; preserve evidence')


if __name__ == '__main__':
    main()
