#!/usr/bin/env python3
"""Compile baseline/candidate engine with a frozen firmware compilation database.

Run inside its pinned SDK image with source/configuration mounted read-only.
Object and per-function stack sizes are not linked-image or runtime acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

NAMES = ('nd_whistle.c', 'nd_cact.c', 'nd_quant.c', 'nd_tokenizer.c', 'nd_ledcmd.c')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile-db', 'baseline-engine', 'candidate-engine', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve earlier evidence; use a new output')
    entries = json.loads(a.compile_db.read_text())
    selected = {}
    for name in NAMES:
        matches = [e for e in entries if Path(e['file']).name == name]
        if len(matches) != 1:
            raise ValueError('Expected exactly one firmware compile command for ' + name)
        selected[name] = matches[0]
    os.umask(0o077)
    a.output.mkdir(parents=True)
    (a.output / 'INCOMPLETE').write_text('Target compilation incomplete.\n')
    variants = {}
    compiler = Path(shlex.split(selected[NAMES[0]]['command'])[0])
    prefix = str(compiler)[:-3]
    version = subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0]
    abi = a.output / 'abi.c'
    abi.write_text('#include "nd_whistle.h"\nunsigned char ack_whistle_state[sizeof(nd_whistle)];\n')
    for label, root in (('baseline', a.baseline_engine), ('candidate', a.candidate_engine)):
        directory = a.output / label
        directory.mkdir()
        cases = []
        for name in (*NAMES, 'abi.c'):
            e = selected[NAMES[0] if name == 'abi.c' else name]
            command = shlex.split(e['command'])
            if '-o' not in command or '-c' not in command:
                raise ValueError('Expected explicit compile-only input/output')
            obj = directory / (name + '.o')
            command[command.index('-o') + 1] = str(obj.resolve())
            source = abi if name == 'abi.c' else root / 'src' / name
            command[command.index('-c') + 1] = str(source.resolve())
            for option in ('-MF', '-MT', '-MQ'):
                if option in command:
                    command[command.index(option) + 1] = str(obj.resolve()) + ('.d' if option == '-MF' else '')
            command.insert(1, '-I' + str((root / 'include').resolve()))
            command.append('-fstack-usage')
            (directory / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
            run = subprocess.run(command, cwd=e['directory'], capture_output=True, text=True, timeout=120)
            (directory / (name + '.stderr')).write_text(run.stderr)
            if run.returncode:
                raise ValueError('Target compilation failed: ' + label + '/' + name)
            size = subprocess.check_output([prefix + 'size', str(obj)], text=True).splitlines()[1].split()
            case = dict(name=name, source_sha256=sha(source), object_sha256=sha(obj),
                        text_bytes=int(size[0]), data_bytes=int(size[1]), bss_bytes=int(size[2]))
            if name == 'abi.c':
                symbols = subprocess.check_output([prefix + 'nm', '-S', str(obj)], text=True)
                symbol = next(line.split() for line in symbols.splitlines() if line.endswith(' ack_whistle_state'))
                case['model_state_target_bytes'] = int(symbol[1], 16)
                header = subprocess.check_output([prefix + 'readelf', '-h', str(obj)], text=True)
                if 'ELF32' not in header or 'Xtensa' not in header:
                    raise ValueError('Expected actual 32-bit Xtensa object')
            frames = []
            for line in obj.with_suffix('.su').read_text().splitlines():
                identity, frame, kind = line.rsplit('\t', 2)
                frames.append(dict(function=identity.rsplit(':', 1)[-1], bytes=int(frame), kind=kind))
            case['stack_frames'] = frames
            cases.append(case)
        variants[label] = cases
    result = dict(schema=1, scope=__doc__, checker_sha256=sha(Path(__file__)),
                  compile_database_sha256=sha(a.compile_db), compiler_version=version,
                  variants=variants, linked_firmware_verified=False,
                  physical_stack_acceptance_verified=False, deployment_permitted=False)
    (a.output / 'target.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'INCOMPLETE').unlink()
    print(json.dumps({k: v for k, v in result.items() if k != 'variants'}), flush=True)


if __name__ == '__main__':
    main()
