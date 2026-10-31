#!/usr/bin/env python3
"""Wait for spare RAM, then run one bounded CPU research job without eviction.

Set host storage/cache variables before invoking. Use a unique report path;
the adjacent log and report preserve admission, command and terminal results.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def available_memory():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))) * 1024


def group_rss(group):
    pages = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            if int(stat[stat.rfind(')') + 2:].split()[2]) == group:
                pages += int((entry / 'statm').read_text().split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return pages * os.sysconf('SC_PAGE_SIZE')


def stop_owned_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may exit while a grandchild remains in the owned group.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--max-rss-gib', type=float, default=4)
    p.add_argument('--reserve-gib', type=float, default=2)
    p.add_argument('--host-floor-gib', type=float, default=1)
    p.add_argument('--wait-seconds', type=float, default=300)
    p.add_argument('--max-seconds', type=float, default=900)
    p.add_argument('--admission-samples', type=int, default=4)
    p.add_argument('--cpus', type=int, default=2)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command[:1] == ['--']:
        a.command = a.command[1:]
    if (not all(math.isfinite(v) for v in (a.max_rss_gib, a.max_seconds, a.reserve_gib, a.host_floor_gib, a.wait_seconds)) or
            not a.command or min(a.max_rss_gib, a.max_seconds, a.admission_samples, a.cpus) <= 0 or min(a.reserve_gib, a.host_floor_gib, a.wait_seconds) < 0):
        p.error('Require a command and positive resource limits')
    if a.report.exists() or a.report.with_suffix('.log').exists():
        p.error('Preserve earlier jobs; use a new report path')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    limit = int(a.max_rss_gib * 1024**3)
    floor = int(a.host_floor_gib * 1024**3)
    launch_floor = max(limit + int(a.reserve_gib * 1024**3), floor)
    status = dict(schema=1, state='waiting', command=a.command, guard_pid=os.getpid(), child_pid=None,
                  rss_limit_bytes=limit, launch_available_memory_bytes=launch_floor, host_available_memory_floor_bytes=floor,
                  cpu_only=True, cpus=a.cpus, nice_increment=10, maximum_job_seconds=a.max_seconds,
                  peak_observed_process_group_rss_bytes=0, admitted=False)
    def write():
        temporary = a.report.with_name(a.report.name + '.pending')
        with temporary.open('x') as out:
            out.write(json.dumps(status, indent=2) + '\n')
        temporary.replace(a.report)
    with a.report.open('x') as out:
        out.write(json.dumps(status, indent=2) + '\n')
    print('Waiting for sustained spare RAM; no workload will be evicted.', flush=True)
    stable = 0
    while True:
        free = available_memory()
        stable = stable + 1 if free >= launch_floor else 0
        if stable >= a.admission_samples:
            break
        if time.monotonic() - started >= a.wait_seconds:
            status.update(state='not-started', reason='insufficient_sustained_memory_headroom', available_memory_bytes=free)
            write()
            return 75
        time.sleep(1)
    cpus = sorted(os.sched_getaffinity(0))[:a.cpus]
    def setup():
        os.sched_setaffinity(0, cpus)
        os.nice(10)
    env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS=str(len(cpus)), MKL_NUM_THREADS=str(len(cpus)))
    job_started = time.monotonic()
    reason = None
    with a.report.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(a.command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=setup)
        try:
            status.update(state='running', admitted=True, child_pid=child.pid, available_at_launch_bytes=free,
                          admission_wait_seconds=round(job_started - started, 3), cpu_affinity=cpus)
            write()
            print('Started owned CPU process', child.pid, flush=True)
            while child.poll() is None:
                rss = group_rss(child.pid)
                status['peak_observed_process_group_rss_bytes'] = max(status['peak_observed_process_group_rss_bytes'], rss)
                if rss > limit:
                    reason = 'owned_process_group_rss_limit'
                elif available_memory() < floor:
                    reason = 'host_available_memory_floor'
                elif time.monotonic() - job_started > a.max_seconds:
                    reason = 'job_time_limit'
                if reason:
                    stop_owned_group(child)
                    break
                time.sleep(.25)
        except BaseException:
            stop_owned_group(child)
            status.update(state='terminated', reason='guard_interrupted', exit_code=child.returncode)
            write()
            raise
        code = child.wait()
        if reason is None and group_rss(child.pid):
            reason = 'owned_descendants_remained_after_exit'
            stop_owned_group(child)
    status.update(state='complete' if code == 0 and reason is None else 'stopped', reason=reason, exit_code=code,
                  job_elapsed_seconds=round(time.monotonic() - job_started, 3))
    write()
    print(json.dumps(status), flush=True)
    return code if code >= 0 and reason is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
