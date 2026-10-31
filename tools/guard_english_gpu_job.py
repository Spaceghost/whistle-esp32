#!/usr/bin/env python3
"""Wait for one idle GPU, then supervise only an owned research process group.

No eviction. NVML metadata is queried through nvidia-smi before CUDA starts.
Limits are sampled withdrawal thresholds, not a hardware memory partition.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from guard_english_job import available_memory, group_rss, stop_owned_group

GIB=1024**3


def snapshot(device):
    query=subprocess.run(['nvidia-smi','--id='+str(device),
        '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    rows=query.stdout.strip().splitlines()
    if len(rows)!=1:
        raise ValueError('Require exactly one GPU')
    index,uuid,total,free,util=(x.strip() for x in rows[0].split(','))
    result=dict(index=int(index),uuid=uuid,total_bytes=int(total)*1024**2,
                free_bytes=int(free)*1024**2,utilization=int(util),processes=[])
    if not uuid.startswith('GPU-') or not 0<=result['utilization']<=100 or not 0<=result['free_bytes']<=result['total_bytes']:
        raise ValueError('Invalid GPU telemetry')
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    for line in processes.stdout.splitlines():
        gpu,pid,used=(x.strip() for x in line.split(','))
        if gpu==uuid:
            if int(pid)<=0 or int(used)<0:
                raise ValueError('Invalid process telemetry')
            result['processes'].append(dict(pid=int(pid),bytes=int(used)*1024**2))
    return result


def group_pids(group):
    result=set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat=(entry/'stat').read_text()
            if int(stat[stat.rfind(')')+2:].split()[2])==group:
                result.add(int(entry.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            pass
    return result


def admission_reason(gpu,host_available,gpu_required,host_required,max_utilization):
    if gpu['processes']:
        return 'other_compute_process_present'
    if gpu['utilization']>max_utilization:
        return 'gpu_utilization_above_admission_limit'
    if gpu['free_bytes']<gpu_required:
        return 'gpu_memory_headroom'
    if host_available<host_required:
        return 'host_memory_headroom'
    return None


def running_reason(gpu,owned,host_available,rss,elapsed,gpu_limit,gpu_floor,rss_limit,host_floor,max_seconds):
    if any(p['pid'] not in owned for p in gpu['processes']):
        return 'other_compute_process_arrived'
    if sum(p['bytes'] for p in gpu['processes'] if p['pid'] in owned)>gpu_limit:
        return 'owned_gpu_memory_limit'
    if gpu['free_bytes']<gpu_floor:
        return 'gpu_free_memory_floor'
    if rss>rss_limit:
        return 'owned_process_group_rss_limit'
    if host_available<host_floor:
        return 'host_available_memory_floor'
    if elapsed>max_seconds:
        return 'job_time_limit'
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--lock-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--gpu-limit-gib',type=float,default=3)
    p.add_argument('--gpu-reserve-gib',type=float,default=2)
    p.add_argument('--jax-pool-gib',type=float,default=2)
    p.add_argument('--max-rss-gib',type=float,default=3)
    p.add_argument('--host-reserve-gib',type=float,default=2)
    p.add_argument('--host-floor-gib',type=float,default=1)
    p.add_argument('--max-utilization',type=int,default=10)
    p.add_argument('--idle-seconds',type=float,default=30)
    p.add_argument('--wait-seconds',type=float,default=300)
    p.add_argument('--max-seconds',type=float,default=900)
    p.add_argument('--cpus',type=int,default=2)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.command[:1]==['--']:
        a.command=a.command[1:]
    values=(a.gpu_limit_gib,a.gpu_reserve_gib,a.jax_pool_gib,a.max_rss_gib,a.host_reserve_gib,a.host_floor_gib,a.idle_seconds,a.wait_seconds,a.max_seconds)
    if (not a.command or not all(math.isfinite(x) and x>=0 for x in values) or
        min(a.gpu_limit_gib,a.jax_pool_gib,a.max_rss_gib,a.max_seconds,a.cpus)<=0 or a.device<0 or
        not 0<=a.max_utilization<=100 or a.jax_pool_gib>=a.gpu_limit_gib):
        p.error('Invalid command, resource limits or pool budget')
    if any(x.exists() for x in (a.report,a.report.with_suffix('.log'),a.report.with_name(a.report.name+'.pending'))):
        p.error('Preserve prior receipts; choose a new report path')
    os.umask(0o077)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.lock_dir.mkdir(parents=True,exist_ok=True)
    cache=a.cache_dir.resolve()
    for name in ('compilation','cuda','triton','xdg'):
        (cache/name).mkdir(parents=True,exist_ok=True)
    started=time.monotonic();child=None;lock=None
    status=dict(schema=1,state='waiting',guard_pid=os.getpid(),child_pid=None,admitted=False,device_index=a.device,
                command=a.command,required_idle_seconds=a.idle_seconds,max_admission_utilization=a.max_utilization,
                gpu_observed_limit_bytes=int(a.gpu_limit_gib*GIB),gpu_free_floor_bytes=int(a.gpu_reserve_gib*GIB),
                host_rss_limit_bytes=int(a.max_rss_gib*GIB),host_free_floor_bytes=int(a.host_floor_gib*GIB),
                jax_pool_bytes=int(a.jax_pool_gib*GIB),limits_are_sampled_not_hardware_partition=True,
                peak_owned_gpu_bytes=0,peak_group_rss_bytes=0,reason=None)
    def write():
        pending=a.report.with_name(a.report.name+'.pending')
        with pending.open('x') as f:
            f.write(json.dumps(status,indent=2)+'\n')
        pending.replace(a.report)
    def interrupted(*_):
        raise KeyboardInterrupt
    old_handler=signal.signal(signal.SIGTERM,interrupted)
    with a.report.open('x') as f:
        f.write(json.dumps(status,indent=2)+'\n')
    try:
        gpu=snapshot(a.device);uuid=gpu['uuid'];fingerprint=hashlib.sha256(uuid.encode()).hexdigest()[:16]
        status['device_fingerprint']=fingerprint
        status['gpu_total_bytes']=gpu['total_bytes']
        if status['jax_pool_bytes']>=gpu['total_bytes']:
            raise ValueError('Pool exceeds physical device memory')
        lock=(a.lock_dir/(fingerprint+'.lock')).open('a+')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            status.update(state='not-started',reason='cooperating_gpu_guard_already_active');write();return 75
        stable_since=None
        print('Waiting for sustained idle GPU and spare RAM; other jobs stay running.',flush=True)
        while True:
            gpu=snapshot(uuid);free=available_memory();now=time.monotonic()
            if gpu['uuid']!=uuid:
                raise ValueError('GPU identity changed during admission')
            reason=admission_reason(gpu,free,int((a.gpu_limit_gib+a.gpu_reserve_gib)*GIB),
                                    int((a.max_rss_gib+a.host_reserve_gib)*GIB),a.max_utilization)
            stable_since=now if reason is None and stable_since is None else stable_since if reason is None else None
            idle=now-stable_since if stable_since is not None else 0
            status.update(reason=reason,observed_gpu_utilization=gpu['utilization'],observed_gpu_free_bytes=gpu['free_bytes'],
                          observed_host_available_bytes=free,observed_compute_processes=len(gpu['processes']),sustained_idle_seconds=idle)
            write()
            if reason is None and idle>=a.idle_seconds:
                break
            if now-started>=a.wait_seconds:
                status.update(state='not-started',admission_wait_seconds=now-started);write();return 75
            time.sleep(1)
        cpus=sorted(os.sched_getaffinity(0))[:a.cpus]
        def setup():
            os.sched_setaffinity(0,cpus);os.nice(10)
        env=dict(os.environ,JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES=uuid,XLA_PYTHON_CLIENT_PREALLOCATE='true',
                 ACK_ENGLISH_GPU_GUARD_PID=str(os.getpid()),ACK_ENGLISH_GPU_GUARD_REPORT=str(a.report.resolve()),
                 XLA_PYTHON_CLIENT_MEM_FRACTION=str(status['jax_pool_bytes']/gpu['total_bytes']),
                 JAX_COMPILATION_CACHE_DIR=str(cache/'compilation'),CUDA_CACHE_PATH=str(cache/'cuda'),
                 TRITON_CACHE_DIR=str(cache/'triton'),XDG_CACHE_HOME=str(cache/'xdg'),
                 OMP_NUM_THREADS=str(len(cpus)),OPENBLAS_NUM_THREADS=str(len(cpus)),MKL_NUM_THREADS=str(len(cpus)))
        # Avoid inherited alternate allocators bypassing the intended BFC pool.
        env.pop('XLA_PYTHON_CLIENT_ALLOCATOR',None);env.pop('TF_GPU_ALLOCATOR',None)
        env.pop('LD_LIBRARY_PATH',None);env.pop('JAX_SKIP_CUDA_CONSTRAINTS_CHECK',None)
        job_started=time.monotonic()
        with a.report.with_suffix('.log').open('x') as log:
            child=subprocess.Popen(a.command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=setup)
            status.update(state='running',admitted=True,child_pid=child.pid,reason=None,cpu_affinity=cpus,
                          admission_wait_seconds=job_started-started)
            write();print('Started owned GPU process',child.pid,flush=True)
            while child.poll() is None:
                gpu=snapshot(uuid);owned=group_pids(child.pid);rss=group_rss(child.pid)
                if gpu['uuid']!=uuid:
                    raise ValueError('GPU identity changed while supervising')
                used=sum(x['bytes'] for x in gpu['processes'] if x['pid'] in owned)
                status['peak_owned_gpu_bytes']=max(status['peak_owned_gpu_bytes'],used)
                status['peak_group_rss_bytes']=max(status['peak_group_rss_bytes'],rss)
                reason=running_reason(gpu,owned,available_memory(),rss,time.monotonic()-job_started,
                    status['gpu_observed_limit_bytes'],status['gpu_free_floor_bytes'],status['host_rss_limit_bytes'],status['host_free_floor_bytes'],a.max_seconds)
                if reason:
                    status['reason']=reason;stop_owned_group(child);break
                write();time.sleep(.5)
            code=child.wait()
            if group_pids(child.pid):
                status['reason']=status['reason'] or 'owned_descendants_remained_after_exit'
                stop_owned_group(child)
        status.update(state='complete' if code==0 and status['reason'] is None else 'stopped',exit_code=code,
                      job_elapsed_seconds=time.monotonic()-job_started)
        write();print(json.dumps(status),flush=True)
        return 0 if status['state']=='complete' else 1
    except BaseException as error:
        if child is not None:
            stop_owned_group(child)
        status.update(state='terminated' if isinstance(error,KeyboardInterrupt) else 'stopped',
                      reason='guard_interrupted' if isinstance(error,KeyboardInterrupt) else 'telemetry_or_guard_error',error_type=type(error).__name__)
        write()
        if isinstance(error,KeyboardInterrupt):
            return 130
        raise
    finally:
        signal.signal(signal.SIGTERM,old_handler)
        if lock is not None:
            lock.close()


if __name__=='__main__':
    raise SystemExit(main())
