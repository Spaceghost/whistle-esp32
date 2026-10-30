#!/usr/bin/env python3
"""Exercise the actual native BPE trainer before a full corpus job; no speech I/O."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import runpy
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Preserve the existing report; choose a new output path')
    import sentencepiece as spm
    module = runpy.run_path(str(Path(__file__).with_name('train_english_tokenizer.py')))
    views = ['This is a sample English tokenizer training sentence number ' + str(i) + '.' for i in range(1000)]
    options = module['training_options'](views)
    output = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(views), model_writer=output,
                                  vocab_size=512, minloglevel=2, **options)
    tokenizer = spm.SentencePieceProcessor(model_proto=output.getvalue())
    for text in ('Levi Ackerman', 'Renée', "Let's make an app."):
        ids = tokenizer.encode(text)
        assert tokenizer.unk_id() not in ids and tokenizer.decode(ids) == text
    report = {'scope': 'Artificial English text API integration check; not speech or corpus fitting',
              'python': sys.version.split()[0], 'sentencepiece': spm.__version__,
              'options': options, 'vocabulary': tokenizer.get_piece_size(),
              'model_sha256': hashlib.sha256(output.getvalue()).hexdigest(),
              'unseen_names_utf8_and_apostrophe_roundtrip': True, 'unknown_tokens': 0}
    if args.output:
        with args.output.open('x') as target:
            target.write(json.dumps(report, indent=2) + '\n')
    print('PASS native tokenizer API: unseen names, UTF-8 and apostrophes round-trip')


if __name__ == '__main__':
    main()
