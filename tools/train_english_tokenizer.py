#!/usr/bin/env python3
"""Train smaller English BPE research vocabularies; never rewrite a Whistle model.

Only verified training-split text fits the vocabulary. Validation/test are scored
after training. New token IDs require neural retraining and runtime/export work.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_splits(root):
    if (root / 'INCOMPLETE').exists():
        raise ValueError('Corpus preparation is incomplete')
    report = json.loads((root / 'corpus.json').read_text())
    if report.get('source_declared_language') != 'en' or report.get('speaker_and_id_and_encoded_audio_disjoint') is not True:
        raise ValueError('Require verified English, disjoint corpus splits')
    splits = {}
    for split in ('train', 'validation', 'test'):
        meta = report['manifests'][split]
        path = (root / meta['file']).resolve()
        if path.parent != root.resolve() or sha(path) != meta['sha256']:
            raise ValueError('Corpus manifest differs from the verified receipt')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows or any(row.get('language') != 'en' or row.get('split') != split or not row.get('text') for row in rows):
            raise ValueError('Missing, foreign-language or mixed-split training metadata')
        splits[split] = rows
    # Recheck leakage before fitting, even if a modified receipt claims success.
    for index, split in enumerate(splits):
        for other in list(splits)[index + 1:]:
            for key in ('speaker', 'id', 'audio_sha256'):
                if {row[key] for row in splits[split]} & {row[key] for row in splits[other]}:
                    raise ValueError('Corpus split leakage before tokenizer fitting')
    return splits, report


def text_views(text):
    # Deterministic casing augmentation of TRAIN text only; source remains intact.
    return list(dict.fromkeys((text, text.lower(), text.capitalize())))


def training_options(views):
    # Byte fallback covers unseen characters. Do not force zero-frequency
    # required_chars: SentencePiece 0.2.2 BPE asserts on those candidates.
    return {'model_type': 'bpe', 'character_coverage': 1.0, 'byte_fallback': True,
            'pad_id': 0, 'eos_id': 1, 'bos_id': 2, 'unk_id': 3,
            'normalization_rule_name': 'identity', 'num_threads': 1,
            'shuffle_input_sentence': False, 'input_sentence_size': 0,
            'max_sentence_length': max(len(x.encode('utf-8')) for x in views) + 1,
            'hard_vocab_limit': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New research artifact directory')
    parser.add_argument('--vocab-sizes', type=int, nargs='+', default=[8192, 4096, 2048])
    args = parser.parse_args()
    if len(set(args.vocab_sizes)) != len(args.vocab_sizes) or any(not 512 <= n <= 8192 for n in args.vocab_sizes):
        parser.error('Use distinct vocabulary sizes between 512 and 8192')
    splits, corpus = load_splits(args.corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    marker = args.output / 'INCOMPLETE'
    marker.write_text('Research tokenizers are not ready. No firmware model is generated.\n')
    training = args.output / 'training-text.txt'
    views = [view for row in splits['train'] for view in text_views(row['text'])]
    training.write_text('\n'.join(views) + '\n')
    import sentencepiece as spm
    options = training_options(views)
    report = {'schema': 1, 'scope': 'English-only training-text tokenizer research; not speech-model training',
              'sentencepiece_version': spm.__version__, 'corpus_sha256': sha(args.corpus / 'corpus.json'),
              'source': corpus['source'], 'license': corpus['license'],
              'training_source': 'train only; original, lowercase and sentence-case text views',
              'training_text_sha256': sha(training), 'options': options,
              'validation_or_test_used_for_fitting': False, 'neural_weights_trained': False,
              'firmware_compatible': False, 'deployment_permitted': False, 'candidates': []}
    for size in args.vocab_sizes:
        prefix = args.output / ('english-bpe-' + str(size))
        spm.SentencePieceTrainer.train(input=str(training), model_prefix=str(prefix), vocab_size=size, **options)
        tokenizer = spm.SentencePieceProcessor(model_file=str(prefix) + '.model')
        scores = {}
        for split, rows in splits.items():
            tokens = words = max_tokens = unknown = 0
            for row in rows:
                # Score all casing views; never feed these held-out strings back to fitting.
                for text in text_views(row['text']):
                    ids = tokenizer.encode(text)
                    if tokenizer.decode(ids) != text:
                        raise ValueError('Text round-trip differs; candidate retained for inspection')
                    tokens += len(ids)
                    words += len(text.split())
                    max_tokens = max(max_tokens, len(ids))
                    unknown += ids.count(tokenizer.unk_id())
            if unknown:
                raise ValueError('Unknown tokens remain despite byte fallback')
            scores[split] = {'tokens': tokens, 'words': words, 'tokens_per_word': tokens / words,
                             'maximum_tokens': max_tokens, 'unknown_tokens': unknown}
        model = Path(str(prefix) + '.model')
        report['candidates'].append({'vocabulary': size, 'model': model.name, 'sha256': sha(model),
            'model_file_bytes': model.stat().st_size, 'scores': scores,
            'estimated_tied_embedding_bytes_saved_at_existing_row_format': (8192 - size) * 264,
            'estimated_logits_bytes_saved': (8192 - size) * 4,
            'estimate_scope': 'Only rows/logits at current geometry; not full-model size or runtime acceptance'})
        print(size, 'pieces;', round(scores['test']['tokens_per_word'], 3), 'held-out tokens/word', flush=True)
    with (args.output / 'tokenizers.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    marker.unlink()
    print('READY: research tokenizers only; new neural model and engine/export compatibility still required', flush=True)


if __name__ == '__main__':
    main()
