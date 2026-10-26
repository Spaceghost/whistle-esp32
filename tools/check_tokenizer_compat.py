#!/usr/bin/env python3
"""Compare two nd_dump builds on deterministic synthetic text, without downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

FIXED = ['', 'a', ' ', '\n', '  leading and trailing  ', 'tabs\tand\nnewlines',
         'Hello Levi. This is a connection test.', 'Levi Ackerman', 'Mikasa Ackerman',
         "I'm okay", 'I’m okay', 'Need you', 'café naïve résumé', '東京 🎉',
         'digits 0123456789 and !@#$%^&*()_+-=[]{}|;:\",./<>?', '▁space marker',
         '<|im_start|>user\nHello Levi<|im_end|>\n<|im_start|>assistant\n',
         '<think>find my words</think>', '<tool_call>[]</tool_call>',
         '<|en|> <|de|> <|fr|> <|es|> <|it|> <|nl|> <|pl|>']


def invoke(binary, model, action, args):
    p = subprocess.run([str(binary), str(model), action, *args], capture_output=True, check=True)
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('model', type=Path)
    ap.add_argument('--random-cases', type=int, default=128)
    a = ap.parse_args()
    if not 0 <= a.random_cases <= 4096: raise ValueError('random cases must be 0..4096')
    baseline, candidate, model = (p.resolve() for p in (a.baseline, a.candidate, a.model))
    rng = random.Random(42)
    alphabet = 'abcde ABCDE0123456789\n\t.,!?é—▁'
    cases = FIXED + [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 120)))
                     for _ in range(a.random_cases)]
    for i, text in enumerate(cases):
        before = invoke(baseline, model, 'tok', [text])
        after = invoke(candidate, model, 'tok', [text])
        if before != after: raise ValueError(f'token IDs differ on synthetic case {i}')
        ids = before.decode().split()
        if invoke(baseline, model, 'detok', ids) != invoke(candidate, model, 'detok', ids):
            raise ValueError(f'decoded bytes differ on synthetic case {i}')
    print(json.dumps({'synthetic_cases': len(cases), 'token_ids_identical': True,
                      'decoded_bytes_identical': True,
                      'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
                      'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()}))


if __name__ == '__main__': main()
