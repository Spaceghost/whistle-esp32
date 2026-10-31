#!/usr/bin/env python3
"""Export a verified fresh English checkpoint for native research, never deployment."""
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


def mark_training_frontend(blob, profile=2):
    """Version the audio manifest explicitly; upstream emits legacy version 1."""
    if profile not in (2, 3):
        raise ValueError('Unknown English frontend profile')
    raw = bytearray(blob)
    header = struct.unpack_from('<48If', raw)
    if header[0] != 0x05E12A84:
        raise ValueError('Not a v4 Whistle container')
    # Walk the canon decoder layout, matching the existing C engine.
    audio = 1 + header[10] * 27 + 9 + 2 + header[31] * 4 + 1
    if audio >= header[1]:
        raise ValueError('Missing audio manifest')
    rec = struct.unpack_from('<BBHIIIIQQII', raw, 196 + header[2] * 4 + audio * 44)
    if rec[0:2] != (2, 1) or rec[3] != 13 or rec[8] != 52 or rec[7] + 52 > len(raw):
        raise ValueError('Unexpected audio manifest geometry')
    values = struct.unpack_from('<13f', raw, rec[7])
    if values[:2] != (3246.0, 1.0):
        raise ValueError('Unknown input frontend; do not silently retag it')
    struct.pack_into('<I', raw, 0, 0x31455741)  # AWE1; older engines must refuse it.
    struct.pack_into('<f', raw, rec[7] + 4, float(profile))
    return bytes(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'needle-source', 'tokenizer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve prior output; choose a fresh directory')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    helper = runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    helper['verify_source'](a.needle_source)
    manifest = helper['checkpoint_manifest'](a.checkpoint)
    metadata = manifest['metadata']
    if not metadata.get('all_parameters_trainable') or metadata.get('initialization') != 'fresh random, no inherited model weights':
        raise ValueError('Expected fresh, full-parameter English research checkpoint')
    signature = metadata['signature']
    if sha(a.tokenizer) != signature['tokenizer_sha256'] or signature['qat_bits'] not in (2, 4):
        raise ValueError('Tokenizer or QAT width differs from the verified checkpoint')
    import english_features
    import english_precision
    profile = signature.get('audio_frontend_profile', 2)
    if profile == 3 and (signature.get('frontend_module_sha256') != sha(Path(english_features.__file__)) or signature.get('precision_module_sha256') != sha(Path(english_precision.__file__)) or signature.get('tied_embedding_bits') != 4):
        raise ValueError('Frontend or precision implementation differs from training')
    if signature['qat_bits'] == 2 and profile != 3:
        raise ValueError('Two-bit body requires its matched stable English training policy')
    whistle, quantize, checkpoints = helper['source_modules'](a.needle_source)
    export = importlib.import_module('_ack_needle.model.export')
    from sentencepiece import sentencepiece_model_pb2
    cfg = whistle.WhistleConfig.from_saved(metadata['config'])
    if cfg.languages != ('en',) or cfg.text not in (2048, 4096, 8192):
        raise ValueError('Expected English-only research vocabulary')
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(a.tokenizer.read_bytes())
    if len(proto.pieces) != cfg.text:
        raise ValueError('Tokenizer/config vocabulary mismatch')
    saved = checkpoints.read_checkpoint(a.checkpoint / 'model.safetensors')
    if helper['canonical'](saved['config']) != helper['canonical'](metadata['config']) or saved['step'] != metadata['step']:
        raise ValueError('Model metadata differs from checkpoint manifest')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'INCOMPLETE').write_text('Research export not complete.\n')
    tokenizer = export.whistle_tokenizer_blob(proto, ('en',))
    print('packing_full_model', flush=True)
    import english_export_math
    blob, tensors = english_export_math.pack_model(export, quantize, saved['params'], cfg.needle(), tokenizer, cfg, signature['qat_bits'])
    blob = mark_training_frontend(blob, profile)
    artifact = a.output / 'english-student-research.cact'
    with artifact.open('xb') as target:
        target.write(blob);target.flush();os.fsync(target.fileno())
    report = {'schema': 1, 'scope': 'Two-step research checkpoint export; not an accepted recognizer',
              'checkpoint_manifest_sha256': sha(a.checkpoint / 'manifest.json'),
              'checkpoint_step': metadata['step'], 'tokenizer_sha256': sha(a.tokenizer),
              'config': metadata['config'], 'training_signature': signature,
              'source_commit': helper['NEEDLE_COMMIT'], 'export_script_sha256': sha(Path(__file__)),
              'packing_math': 'pinned trainer JAX float32, codebook choice and FP16 norm rounding',
              'packing_module_sha256': sha(Path(english_export_math.__file__)),
              'bytes': len(blob), 'sha256': sha(artifact), 'tensors': tensors,
              'text_vocabulary': cfg.text, 'english_prefix_id': cfg.text, 'embedding_rows': cfg.vocab,
              'body_weight_bits': signature['qat_bits'], 'tied_embedding_bits': 4, 'audio_frontend_profile': profile, 'native_format': 'AWE1',
              'native_validation_complete': False, 'neural_accuracy_verified': False,
              'deployment_permitted': False}
    (a.output / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    (a.output / 'INCOMPLETE').unlink()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
