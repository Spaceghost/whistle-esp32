#!/usr/bin/env python3
"""Score verified held-out English audio with a frozen native engine and model.

No training, keyword hints, speech trimming, resampling, deployment or CUDA.
Engine failures leave INCOMPLETE; a subset is never labelled a complete split.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
import unicodedata
import wave

from train_english_student import audio_path, sha
from train_english_tokenizer import load_splits

NORMALIZATION = 'nfkc-casefold-punctuation-spaces-internal-apostrophes-v1'


def words(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('\u2019', "'").replace('\u2018', "'")
    normalized = []
    for i, char in enumerate(text):
        internal_quote = char == "'" and 0 < i < len(text)-1 and text[i-1].isalnum() and text[i+1].isalnum()
        normalized.append(char if char.isalnum() or internal_quote else ' ')
    return ''.join(normalized).split()


def word_errors(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(ref != hyp)))
        previous = current
    return previous[-1]


def select_rows(splits, split, limit, seed):
    if split not in ('validation', 'test') or limit < 0 or seed < 0:
        raise ValueError('Evaluation requires a held-out split and nonnegative selection settings')
    rows = list(splits[split])
    ids = [row.get('id') for row in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing or repeated held-out clip identity')
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def source_coverage(corpus, split, retained, selected):
    stats = corpus.get('splits', {}).get(split)
    maximum = corpus.get('maximum_seconds')
    result = dict(known=stats is not None, maximum_seconds=maximum,
                  retained_utterances=retained, selected_utterances=selected,
                  source_utterances=None, excluded_over_limit=None, complete_original_split=None)
    if stats is None:
        return result
    values = [stats.get(key) for key in ('source_utterances', 'retained', 'excluded_over_limit')]
    if (any(type(value) is not int or value < 0 for value in values) or
            values[1] != retained or values[0] != values[1]+values[2]):
        raise ValueError('Source coverage counts differ from the retained manifest')
    if maximum is not None and (type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0):
        raise ValueError('Invalid source preparation duration limit')
    result.update(source_utterances=values[0], excluded_over_limit=values[2],
                  complete_original_split=selected==values[0] and values[2]==0)
    return result


def freeze(source, target, executable=False):
    checksum = sha(source)
    shutil.copyfile(source, target)
    if sha(target) != checksum or sha(source) != checksum:
        raise ValueError('Evaluation input changed during freezing')
    target.chmod(0o500 if executable else 0o400)
    return checksum


def make_wav(row, source_root, local_root, target):
    import soundfile as sf
    path = audio_path(row, source_root, local_root)
    if sha(path) != row['audio_sha256']:
        raise ValueError('Held-out audio hash differs')
    info = sf.info(path)
    if (info.samplerate, info.channels, info.frames, info.subtype) != (16000, 1, row['samples'], 'PCM_16'):
        raise ValueError('Held-out audio geometry differs; no truncation or resampling allowed')
    audio, rate = sf.read(path, dtype='int16', always_2d=True)
    if rate != 16000 or audio.shape != (row['samples'], 1):
        raise ValueError('Decoded held-out audio geometry differs')
    with wave.open(str(target), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(audio.astype('<i2', copy=False).tobytes())
    return sha(target)


def transcribe(engine, model, wav, depth, timeout):
    result = subprocess.run([str(engine), str(model), str(wav), '--lang', 'en', '--depth', str(depth)],
        capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=timeout,
        env=dict(os.environ, NEEDLE_TELEMETRY='0'))
    if result.returncode != 0:
        raise ValueError('Native engine failed with exit '+str(result.returncode)+': '+result.stderr[-2048:])
    if len(result.stdout) > 65536 or '\0' in result.stdout:
        raise ValueError('Native engine output is invalid or exceeds its bound')
    return result.stdout.strip()


def write_json(path, value):
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('w') as file:
        file.write(json.dumps(value, indent=2)+'\n')
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def read_evaluation(root):
    if (root/'INCOMPLETE').exists():
        raise ValueError('Held-out evaluation is incomplete')
    report = json.loads((root/'evaluation.json').read_text())
    if (report.get('complete') is not True or report['clips'] != report['selected_clips'] or
            report['clips'] <= 0 or report['reference_words'] <= 0 or
            report['wer'] != report['word_errors']/report['reference_words']):
        raise ValueError('Held-out evaluation totals differ')
    for path, key in ((root/'utterances.jsonl', 'utterances_sha256'),
                      (root/'inputs/native-engine', 'engine_sha256'),
                      (root/'inputs/model.cact', 'model_sha256')):
        if sha(path) != report[key]:
            raise ValueError('Held-out evaluation evidence changed')
    return report


def evaluate(args):
    os.umask(0o077)
    if args.limit < 0 or args.seed < 0 or args.depth < 2 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Invalid evaluation bounds')
    splits, corpus = load_splits(args.corpus)
    rows = select_rows(splits, args.split, args.limit, args.seed)
    coverage = source_coverage(corpus, args.split, len(splits[args.split]), len(rows))
    args.output.mkdir(parents=True, exist_ok=False)
    marker = args.output/'INCOMPLETE'
    marker.write_text('Held-out evaluation is not complete.\n')
    inputs = args.output/'inputs'; inputs.mkdir()
    engine, model = inputs/'native-engine', inputs/'model.cact'
    completed = total_errors = total_words = exact = 0
    started = time.monotonic()
    clip_id = None
    try:
        engine_sha = freeze(args.engine, engine, executable=True)
        model_sha = freeze(args.model, model)
        plan = dict(schema=1, split=args.split, split_clips=len(splits[args.split]), selected_clips=len(rows),
            full_split=len(rows)==len(splits[args.split]), limit=args.limit, seed=args.seed,
            source_coverage=coverage,
            selected_ids_sha256=hashlib.sha256(json.dumps([r['id'] for r in rows], separators=(',', ':')).encode()).hexdigest(),
            corpus_sha256=sha(args.corpus/'corpus.json'), corpus_source=corpus.get('source'),
            manifest_sha256=corpus['manifests'][args.split]['sha256'], engine_sha256=engine_sha,
            model_sha256=model_sha, scorer_sha256=sha(Path(__file__)), normalization=NORMALIZATION,
            depth=args.depth, language='en', keyword_hints=False, audio_truncated=False,
            resampled=False, training_performed=False, deployment_permitted=False)
        write_json(args.output/'plan.json', plan)
        with (args.output/'utterances.jsonl').open('x') as ledger:
            with tempfile.TemporaryDirectory(prefix='wav-', dir=args.output) as directory:
                wav = Path(directory)/'clip.wav'
                for row in rows:
                    clip_id = row['id']
                    reference = words(row['text'])
                    if not reference:
                        raise ValueError('Empty normalized reference')
                    wav_sha = make_wav(row, args.audio_source_root, args.audio_local_root, wav)
                    began = time.monotonic()
                    hypothesis = transcribe(engine, model, wav, args.depth, args.timeout)
                    errors = word_errors(reference, words(hypothesis))
                    elapsed = time.monotonic()-began
                    ledger.write(json.dumps(dict(id=clip_id, reference=row['text'], hypothesis=hypothesis,
                        audio_sha256=row['audio_sha256'], wav_sha256=wav_sha, samples=row['samples'],
                        reference_words=len(reference), word_errors=errors, engine_seconds=elapsed))+'\n')
                    ledger.flush()
                    completed += 1; total_errors += errors; total_words += len(reference)
                    exact += words(hypothesis)==reference
                    write_json(args.output/'progress.json', dict(completed=completed, selected=len(rows),
                        partial_wer=total_errors/total_words, complete=False))
            ledger.flush()
            os.fsync(ledger.fileno())
        if sha(engine) != engine_sha or sha(model) != model_sha:
            raise ValueError('Frozen engine or model changed during evaluation')
        report = dict(plan, complete=True, clips=completed, reference_words=total_words,
                      word_errors=total_errors, wer=total_errors/total_words, exact_utterances=exact,
                      elapsed_seconds=time.monotonic()-started, recognition_accuracy_accepted=False,
                      utterances_sha256=sha(args.output/'utterances.jsonl'))
        write_json(args.output/'evaluation.json', report)
        marker.unlink()
        descriptor = os.open(args.output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return read_evaluation(args.output)
    except Exception as exc:
        write_json(args.output/'failure.json', dict(complete=False, clip_id=clip_id, completed=completed,
            exception=type(exc).__name__, error=str(exc), deployment_permitted=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('corpus', 'model', 'engine', 'audio-source-root', 'audio-local-root', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps({key: report[key] for key in ('split', 'clips', 'full_split', 'source_coverage', 'wer', 'complete', 'deployment_permitted')}))


if __name__ == '__main__':
    main()
