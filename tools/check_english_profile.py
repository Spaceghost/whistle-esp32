#!/usr/bin/env python3
"""Compare original/pruned Whistle inference on a synthetic WAV (host only)."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from whistle_english import english_bytes, KEEP, OLD


def run(binary, model, wav, extra=()):
    return subprocess.run([str(binary), str(model), str(wav), '--lang', 'en', *extra],
                          capture_output=True, text=True, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('binary', type=Path)
    ap.add_argument('original', type=Path)
    ap.add_argument('english', type=Path)
    ap.add_argument('wav', type=Path)
    a = ap.parse_args()
    binary, original, english, wav = (p.resolve() for p in (a.binary, a.original, a.english, a.wav))
    if english_bytes(original.read_bytes()) != english.read_bytes():
        raise ValueError('candidate is not the exact conditioning-row pruning')
    before = run(binary, original, wav)
    after = run(binary, english, wav)
    def token_ids(result):
        for line in result.stderr.splitlines():
            if line.startswith('tokens '): return [int(t) for t in line.split()[1:]]
        raise ValueError('missing decoder token IDs')
    tokens = token_ids(before)
    if tokens != token_ids(after) or before.stdout != after.stdout:
        raise ValueError('greedy transcript differs; candidate needs accuracy review')
    # Teacher forcing preserves history even if an altered output head would
    # change greedy selection. Compare EVERY retained logit, not only winners.
    prefix = [2, 8192] + tokens
    with tempfile.TemporaryDirectory(prefix='whistle-english-') as tmp:
        tmp = Path(tmp)
        for name, model in (('original', original), ('english', english)):
            run(binary, model, wav, ['--enc-out', str(tmp / (name+'.enc')),
                '--logits-out', str(tmp / (name+'.logits')), '--tokens', ','.join(map(str, prefix))])
        if (tmp/'original.enc').read_bytes() != (tmp/'english.enc').read_bytes():
            raise ValueError('encoder output differs')
        original_lg, english_lg = ((tmp/(name+'.logits')).read_bytes() for name in ('original','english'))
        if len(original_lg) != len(prefix)*OLD*4 or len(english_lg) != len(prefix)*KEEP*4:
            raise ValueError('unexpected logit dimensions')
        for i in range(len(prefix)):
            if original_lg[i*OLD*4:(i*OLD+KEEP)*4] != english_lg[i*KEEP*4:(i+1)*KEEP*4]:
                raise ValueError(f'retained logits differ at prefix position {i}')
    print(json.dumps({'encoder_bit_identical': True, 'retained_logits_bit_identical': True,
                      'prefix_positions': len(prefix), 'retained_rows': KEEP,
                      'transcript_token_ids_identical': True, 'transcript_identical': True}))


if __name__ == '__main__': main()
