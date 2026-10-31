#!/usr/bin/env python3
"""Run actual-model decoder and malformed-input checks under proven sanitizers.

No JAX, training, numerical-parity acceptance or device access.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'encoder', 'tokens', 'sanitizer-proof', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if a.output.exists():
        p.error('Preserve earlier evidence; choose a new output')
    proof = json.loads(a.sanitizer_proof.read_text())
    if proof.get('all_passed') is not True or proof.get('checker_sha256') != sha(Path(__file__).with_name('check_sanitizer_runtime.py')):
        raise ValueError('Require current successful ASan and UBSan fault canaries')
    symbols = subprocess.check_output(['nm', str(a.binary)], text=True)
    if '__asan_init' not in symbols or '__ubsan_' not in symbols:
        raise ValueError('Decoder binary lacks actual ASan/UBSan instrumentation')
    export = json.loads((a.model.parent / 'export.json').read_text())
    if sha(a.model) != export['sha256'] or (a.model.parent / 'INCOMPLETE').exists():
        raise ValueError('Require the complete hash-verified research export')
    cfg = export['config']
    encoder, tokens = a.encoder.read_bytes(), a.tokens.read_bytes()
    row = cfg['dim'] * 4
    vocab = cfg['text'] + len(cfg['languages'])
    if (not encoder or len(encoder) % row or len(encoder) // row > 400 or
            not tokens or len(tokens) % 4 or len(tokens) // 4 > cfg['max_tokens'] or
            not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', encoder)) or
            any(x[0] >= vocab for x in struct.iter_unpack('<I', tokens))):
        raise ValueError('Invalid positive-control inputs')
    os.umask(0o077)
    a.output.mkdir(parents=True)
    (a.output / 'INCOMPLETE').write_text('Decoder sanitizer checks incomplete.\n')
    inputs = a.output / 'inputs'
    inputs.mkdir()
    cases = [('valid', encoder, tokens, 0),
             ('nan', struct.pack('<f', float('nan')) + encoder[4:], tokens, 1),
             ('misaligned', encoder + b'\0', tokens, 1),
             ('empty', b'', tokens, 1),
             ('out-of-vocabulary', encoder, struct.pack('<I', vocab), 1),
             ('empty-tokens', encoder, b'', 1),
             ('misaligned-tokens', encoder, tokens + b'\0', 1)]
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1')
    records = []

    def run_case(name, enc, tok, output, expected):
        run = subprocess.run([str(a.binary.resolve()), str(a.model.resolve()),
                              str(enc.resolve()), str(tok.resolve()), str(output.resolve())],
                             env=env, capture_output=True, text=True, timeout=120)
        (a.output / (name + '.stderr')).write_text(run.stderr)
        clean = not any(s in run.stderr for s in ('Sanitizer', 'runtime error:'))
        record = dict(case=name, exit_code=run.returncode, expected=expected,
                      no_sanitizer_diagnostic=clean)
        records.append(record)
        print(json.dumps(record), flush=True)
        if run.returncode != expected or not clean:
            raise ValueError('Decoder safety check failed; preserve diagnostics: ' + name)

    for name, enc, tok, expected in cases:
        ef, tf = inputs / (name + '.f32'), inputs / (name + '.u32')
        ef.write_bytes(enc)
        tf.write_bytes(tok)
        output = a.output / name
        output.mkdir()
        run_case(name, ef, tf, output, expected)
        if expected and inventory(output):
            raise ValueError('Malformed input wrote diagnostic outputs')
    before = inventory(a.output / 'valid')
    if not before or 'logits.f32' not in before:
        raise ValueError('Positive control did not complete decoder execution')
    run_case('existing-output-preserved', inputs / 'valid.f32', inputs / 'valid.u32', a.output / 'valid', 1)
    if inventory(a.output / 'valid') != before:
        raise ValueError('Repeated run changed existing outputs')
    result = dict(schema=1, scope=__doc__, cases=records, all_passed=True,
                  existing_output_inventory_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                  existing_outputs_preserved=True, binary_sha256=sha(a.binary),
                  checker_sha256=sha(Path(__file__)), model_sha256=sha(a.model),
                  encoder_sha256=sha(a.encoder), tokens_sha256=sha(a.tokens),
                  sanitizer_proof_sha256=sha(a.sanitizer_proof),
                  numerical_parity_verified=False, deployment_permitted=False)
    (a.output / 'safety.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'INCOMPLETE').unlink()


if __name__ == '__main__':
    main()
