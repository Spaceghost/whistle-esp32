#!/usr/bin/env python3
"""Prepare source-verified, speaker-disjoint LibriSpeech splits for Whistle work.

No training, downloads, resampling or transcript truncation. Archives and outputs
must live on the host's approved data volume. Partial outputs are preserved.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

SPLITS = {
    'train': ('train-clean-100', '2a93770f6d5c6c964bc36631d331a522'),
    'validation': ('dev-clean', '42e2234ba48799c1f50f24a7926300a1'),
    'test': ('test-clean', '32fa31d27d2e1cad72775fee3f4849a9'),
}
SOURCE = 'https://www.openslr.org/12/'
IDENTIFIER = re.compile(r'([0-9]+)-([0-9]+)-([0-9]+)')


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def flac_samples(path):
    with path.open('rb') as source:
        head = source.read(42)
    if len(head) != 42 or head[:4] != b'fLaC' or head[4] & 127 or int.from_bytes(head[5:8], 'big') != 34:
        raise ValueError(f'Invalid FLAC STREAMINFO: {path.name}')
    packed = int.from_bytes(head[18:26], 'big')
    rate, channels, bits = packed >> 44, ((packed >> 41) & 7) + 1, ((packed >> 36) & 31) + 1
    samples = packed & ((1 << 36) - 1)
    if (rate, channels, bits) != (16000, 1, 16) or not samples:
        raise ValueError(f'Expected nonempty mono 16 kHz PCM16 source: {path.name}')
    return samples


def extract_split(archive, name, destination):
    """Copy regular source files only, without following archive links."""
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as source:
        for entry in source:
            part = PurePosixPath(entry.name)
            if part.is_absolute() or '..' in part.parts or entry.issym() or entry.islnk():
                raise ValueError('Unsafe archive member')
            if not entry.isdir() and not entry.isfile():
                raise ValueError('Unsupported archive member type')
            if entry.isfile() and part == PurePosixPath('LibriSpeech/LICENSE.TXT'):
                target = destination / 'SOURCE-LICENSE.txt'
            elif len(part.parts) >= 3 and part.parts[:2] == ('LibriSpeech', name):
                target = destination.joinpath(*part.parts[2:])
            else:
                continue
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(entry) as data, target.open('xb') as output:
                shutil.copyfileobj(data, output)


def records_for(root, split, max_seconds):
    records, speakers, ids, audio_hashes = [], set(), set(), set()
    excluded_long = total = 0
    seen_audio = set()
    for transcript in sorted(root.rglob('*.trans.txt')):
        for line in transcript.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            identity, text = line.split(maxsplit=1)
            match = IDENTIFIER.fullmatch(identity)
            if not match or not text.strip() or identity in ids:
                raise ValueError('Malformed or duplicate utterance identity')
            speaker, chapter, _ = match.groups()
            if transcript.parent.name != chapter or transcript.parent.parent.name != speaker:
                raise ValueError('Utterance identity differs from speaker/chapter path')
            ids.add(identity)
            speakers.add(speaker)
            audio = transcript.parent / (identity + '.flac')
            samples = flac_samples(audio)
            audio_hash = digest(audio)
            audio_hashes.add(audio_hash)
            seen_audio.add(audio.resolve())
            total += 1
            if samples > max_seconds * 16000:
                excluded_long += 1
                continue
            records.append({'id': identity, 'audio': str(audio.resolve()), 'text': text,
                            'language': 'en', 'speaker': speaker, 'split': split,
                            'samples': samples, 'seconds': samples / 16000,
                            'audio_sha256': audio_hash})
    if not records:
        raise ValueError(f'No eligible source utterances in {split}')
    if {f.resolve() for f in root.rglob('*.flac')} != seen_audio:
        raise ValueError('Audio/transcript pairing is incomplete')
    stats = {'source_utterances': total, 'retained': len(records), 'excluded_over_limit': excluded_long,
             'speakers': len(speakers), 'retained_hours': sum(r['samples'] for r in records) / 16000 / 3600,
             'watch_eight_second_utterances': sum(r['samples'] <= 128000 for r in records)}
    return records, {'speakers': speakers, 'ids': ids, 'audio_hashes': audio_hashes}, stats


def assert_disjoint(groups):
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            for field in ('speakers', 'ids', 'audio_hashes'):
                if groups[name][field] & groups[other][field]:
                    raise ValueError(f'{field} overlap between {name} and {other}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True, help='New extracted corpus directory')
    parser.add_argument('--output', type=Path, required=True, help='New manifest/evidence directory')
    parser.add_argument('--max-seconds', type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 30:
        parser.error('Clip limit must be finite, positive and at most 30 seconds')
    if args.output.exists() or args.audio_dir.exists():
        parser.error('Preserve previous outputs; choose new audio and manifest directories')
    archives = []
    for split, (name, expected_md5) in SPLITS.items():
        archive = args.archive_dir / (name + '.tar.gz')
        if digest(archive, 'md5') != expected_md5:
            parser.error('Archive differs from the official source checksum: ' + name)
        archives.append({'split': split, 'name': name, 'file': archive.name,
                         'upstream_md5': expected_md5, 'sha256': digest(archive),
                         'bytes': archive.stat().st_size})
    args.output.mkdir(parents=True, exist_ok=False)
    incomplete = args.output / 'INCOMPLETE'
    incomplete.write_text('Preparation is unfinished. Do not train or score these outputs.\n')
    args.audio_dir.mkdir(parents=True, exist_ok=False)
    prepared, groups, statistics = {}, {}, {}
    for source in archives:
        name, split = source['name'], source['split']
        root = args.audio_dir / name
        extract_split(args.archive_dir / source['file'], name, root)
        prepared[split], groups[split], statistics[split] = records_for(root, split, args.max_seconds)
        print(split, statistics[split]['retained'], 'eligible utterances', flush=True)
    assert_disjoint(groups)
    manifests = {}
    for split, records in prepared.items():
        path = args.output / (split + '.jsonl')
        with path.open('x') as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
        manifests[split] = {'file': path.name, 'sha256': digest(path)}
    report = {'schema': 1, 'source': SOURCE, 'license': 'CC-BY-4.0',
              'attribution': 'LibriSpeech: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur (2015)',
              'source_declared_language': 'en', 'text_case': 'source preserved',
              'maximum_seconds': args.max_seconds, 'audio_or_text_truncated': False,
              'flac_streaminfo_verified': True, 'all_audio_fully_decoded': False,
              'speaker_and_id_and_encoded_audio_disjoint': True,
              'archives': archives, 'manifests': manifests, 'splits': statistics,
              'training_run': False, 'monolingual_model_proven': False}
    with (args.output / 'corpus.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    incomplete.unlink()
    print('READY: speaker-disjoint English source manifests; no model trained', flush=True)


if __name__ == '__main__':
    main()
