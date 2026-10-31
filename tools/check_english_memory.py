#!/usr/bin/env python3
"""Measure native requested heap on synthetic silence; no device acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'export-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve earlier evidence; choose a new output')
    export = json.loads((a.export_root / 'export.json').read_text())
    model = a.export_root / 'english-student-research.cact'
    if ((a.export_root / 'INCOMPLETE').exists() or sha(model) != export['sha256'] or
            model.stat().st_size != export['bytes'] or export['deployment_permitted'] is not False or
            export['audio_frontend_profile'] != 3 or export['config']['languages'] != ['en']):
        raise ValueError('Need the complete hash-verified English profile-3 research export')
    os.umask(0o077)
    a.output.mkdir(parents=True)
    (a.output / 'INCOMPLETE').write_text('Native memory measurements incomplete.\n')
    cases = []
    for seconds in (1, 4, 8):
        run = subprocess.run([str(a.binary.resolve()), str(model.resolve()), str(seconds)],
                             capture_output=True, text=True, timeout=300)
        (a.output / f'{seconds}s.stderr').write_text(run.stderr)
        (a.output / f'{seconds}s.stdout').write_text(run.stdout)
        if run.returncode:
            raise ValueError('Native memory probe failed; retain its output')
        case = json.loads(run.stdout)
        if (case['seconds'] != seconds or case['passed'] is not True or case['final_live_bytes'] or
                case['allocation_failures'] or not 0 <= case['decode_tokens'] <= 128):
            raise ValueError('Incomplete inference or tracked allocation leak')
        case['requested_heap_peak_bytes'] = max(case[k] for k in
            ('model_open_peak_bytes', 'encode_peak_bytes', 'decode_peak_bytes'))
        # Current Watch owns two fixed eight-second float capture buffers.
        # This arithmetic excludes the UI/network/app heaps and allocator costs.
        case['heap_peak_plus_two_128000_sample_capture_buffers_bytes'] = case['requested_heap_peak_bytes'] + 2 * 128000 * 4
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = dict(schema=1, scope=__doc__, model_sha256=sha(model),
                  export_receipt_sha256=sha(a.export_root / 'export.json'),
                  binary_sha256=sha(a.binary), checker_sha256=sha(Path(__file__)),
                  cases=cases, stimulus='synthetic digital silence', decode_token_cap=128,
                  model_weights_mmap_not_counted_as_heap=True,
                  excludes=['allocator metadata and fragmentation', 'stack and task memory',
                            'UI/network/app heaps', 'physical PSRAM/internal heap placement'],
                  all_tracked_allocations_released=True, watch_memory_acceptance_verified=False,
                  neural_accuracy_verified=False, deployment_permitted=False)
    (a.output / 'memory.json').write_text(json.dumps(report, indent=2) + '\n')
    (a.output / 'INCOMPLETE').unlink()


if __name__ == '__main__':
    main()
