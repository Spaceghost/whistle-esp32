#!/usr/bin/env python3
"""Audit full local English audio and GPU metadata without loading JAX.

No remote access, downloads, model allocation or training launch. Missing audio
is listed for later staging, never silently reduced to the small cached probe.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import subprocess


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_audio(splits, mapper, source_root, local_root, missing, corrupt):
    summary={}
    for split,rows in splits.items():
        stats=dict(required=len(rows),present=0,missing=0,hash_verified=0,hash_mismatch=0,
                   bytes_present=0,samples_required=0,samples_verified=0)
        for row in rows:
            path=mapper(row,source_root,local_root)
            relative=str(PurePosixPath(row['audio']).relative_to(PurePosixPath(str(source_root))))
            record=dict(split=split,id=row['id'],relative_audio=relative,sha256=row['audio_sha256'],samples=row['samples'])
            stats['samples_required']+=row['samples']
            if not path.is_file():
                stats['missing']+=1
                missing.write(json.dumps(record)+'\n')
                continue
            stats['present']+=1
            stats['bytes_present']+=path.stat().st_size
            if sha(path)!=row['audio_sha256']:
                stats['hash_mismatch']+=1
                corrupt.write(json.dumps(record)+'\n')
                continue
            stats['hash_verified']+=1
            stats['samples_verified']+=row['samples']
        stats['hours_required']=stats['samples_required']/16000/3600
        stats['hours_verified']=stats['samples_verified']/16000/3600
        stats['complete']=stats['hash_verified']==stats['required'] and stats['required']>0
        summary[split]=stats
    return summary


def gpu_snapshot():
    # Query metadata only; no CUDA/JAX context is created. Join UUIDs locally,
    # but omit UUIDs and other processes' identities from the retained report.
    query=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version','--format=csv,noheader,nounits']
    try:
        rows=subprocess.run(query,check=True,capture_output=True,text=True,timeout=10).stdout.splitlines()
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
                                 check=True,capture_output=True,text=True,timeout=10).stdout.splitlines()
        usage={}
        for line in processes:
            uuid,memory=(x.strip() for x in line.split(','))
            entry=usage.setdefault(uuid,{'compute_processes':0,'compute_memory_mib':0})
            entry['compute_processes']+=1
            entry['compute_memory_mib']+=int(memory)
        devices=[]
        for line in rows:
            index,uuid,name,total,free,util,driver=(x.strip() for x in line.split(','))
            devices.append(dict(index=int(index),name=name,total_memory_mib=int(total),free_memory_mib=int(free),
                                utilization_percent=int(util),driver_version=driver,
                                **usage.get(uuid,{'compute_processes':0,'compute_memory_mib':0})))
        return dict(query_complete=True,devices=devices,sustained_idle_admission_verified=False)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        return dict(query_complete=False,error_type=type(error).__name__,devices=[],sustained_idle_admission_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','tokenizers','needle-source','audio-source-root','audio-local-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--vocabulary',type=int,choices=(2048,4096,8192),default=2048)
    a=p.parse_args()
    os.umask(0o077)
    if a.output.exists():
        p.error('Preserve evidence; use a fresh audit output directory')
    helper=runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    source_hash=helper['verify_source'](a.needle_source)
    splits,tokenizer,inputs=helper['research_inputs'](a.corpus,a.tokenizers,a.vocabulary)
    a.output.mkdir(parents=True)
    (a.output/'INCOMPLETE').write_text('Full audio audit incomplete.\n')
    with (a.output/'missing-audio.jsonl').open('x') as missing,(a.output/'corrupt-audio.jsonl').open('x') as corrupt:
        audio=audit_audio(splits,helper['audio_path'],a.audio_source_root,a.audio_local_root,missing,corrupt)
    packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()
              if any(x in d.metadata['Name'].lower() for x in ('jax','cuda','cudnn'))}
    result=dict(schema=1,scope=__doc__,audit_complete=True,full_local_audio_hash_verified=all(v['complete'] for v in audio.values()),
                split_audio=audio,input_receipts=inputs,needle_source_receipt_sha256=source_hash,
                tokenizer_sha256=sha(tokenizer),auditor_sha256=sha(Path(__file__)),
                trainer_sha256=sha(Path(__file__).with_name('train_english_student.py')),
                packages=packages,gpu_snapshot=gpu_snapshot(),jax_cuda_plugin_installed=any(x in packages for x in ('jax-cuda12-plugin','jax-cuda13-plugin')),
                gpu_backend_verified=False,gpu_launch_permitted=False,
                missing_audio_list_sha256=sha(a.output/'missing-audio.jsonl'),corrupt_audio_list_sha256=sha(a.output/'corrupt-audio.jsonl'),
                training_started=False,deployment_permitted=False)
    (a.output/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'INCOMPLETE').unlink()
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
