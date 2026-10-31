#!/usr/bin/env python3
"""Require actual ASan and UBSan fault detection, not accepted compiler flags."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess

CASES = {
    'heap': ('address', 'AddressSanitizer: heap-buffer-overflow',
             '#include <stdlib.h>\nvoid *__real_malloc(size_t);\n'
             'void *__wrap_malloc(size_t n){return __real_malloc(n);}\n'
             'int main(int argc,char **argv){volatile char *p=malloc(8);'
             'p[argc+8]=42;free((void*)p);return 0;}\n'),
    'overflow': ('undefined', 'signed integer overflow',
                 'int main(int argc,char **argv){volatile int x=2147483647;return x+argc;}\n')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cc', type=Path, required=True)
    p.add_argument('--library-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if a.output.exists():
        p.error('Preserve prior results; choose a new output')
    os.umask(0o077)
    a.output.mkdir(parents=True)
    (a.output / 'INCOMPLETE').write_text('Sanitizer detection checks incomplete.\n')
    records = []
    for name, (sanitizer, expected, source) in CASES.items():
        cfile = a.output / (name + '.c')
        binary = a.output / name
        cfile.write_text(source)
        command = [str(a.cc.resolve()), '-O1', '-fsanitize=' + sanitizer,
                   '-fno-omit-frame-pointer']
        if name == 'heap':
            command.append('-Wl,--wrap=malloc')
        if a.library_dir:
            directory = str(a.library_dir.resolve())
            command += ['-L' + directory, '-Wl,-rpath,' + directory]
        compiled = subprocess.run(command + [str(cfile), '-o', str(binary)],
                                  capture_output=True, text=True, timeout=120)
        (a.output / (name + '.compile.stderr')).write_text(compiled.stderr)
        record = dict(case=name, sanitizer=sanitizer, source_sha256=sha(cfile),
                      compile_exit_code=compiled.returncode, expected_error_reported=False)
        if compiled.returncode == 0:
            env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                       UBSAN_OPTIONS='halt_on_error=1')
            run = subprocess.run([str(binary.resolve())], env=env, capture_output=True,
                                 text=True, timeout=30)
            (a.output / (name + '.stderr')).write_text(run.stderr)
            record.update(exit_code=run.returncode, binary_sha256=sha(binary),
                          expected_error_reported=run.returncode != 0 and expected in run.stderr)
        records.append(record)
    result = dict(schema=1, scope=__doc__, checker_sha256=sha(Path(__file__)),
                  compiler_command=str(a.cc.resolve()), cases=records,
                  all_passed=all(r['expected_error_reported'] for r in records))
    (a.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'INCOMPLETE').unlink()
    print(json.dumps(result), flush=True)
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
