#!/usr/bin/env python3
"""Stage exact pinned LibriSpeech archives locally; no Bak access or training.

Rate-limited HTTPS, retained partial downloads, exact size/MD5/SHA checks and
regular-file-only extraction. Every output/evidence directory must be new.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time


def verify(path, record):
    if path.stat().st_size!=record['bytes']:
        raise ValueError('Archive size mismatch: '+record['file'])
    hashes={k:hashlib.new(k) for k in ('md5','sha256')}
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            for digest in hashes.values():
                digest.update(chunk)
    if hashes['md5'].hexdigest()!=record['upstream_md5'] or hashes['sha256'].hexdigest()!=record['sha256']:
        raise ValueError('Archive checksum mismatch; preserve bytes: '+record['file'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','archive-cache','audio-output','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rate-limit',default='8M')
    a=p.parse_args()
    os.umask(0o077)
    if a.output.exists() or a.audio_output.exists():
        p.error('Preserve existing evidence/audio; choose new output directories')
    prep=runpy.run_path(str(Path(__file__).with_name('prepare_english_corpus.py')))
    corpus=json.loads((a.corpus/'corpus.json').read_text())
    archives=corpus['archives']
    if len(archives)!=3 or {x['split'] for x in archives}!=set(prep['SPLITS']):
        raise ValueError('Expected exactly three verified corpus archives')
    for record in archives:
        name,md5=prep['SPLITS'][record['split']]
        if record['name']!=name or record['file']!=name+'.tar.gz' or record['upstream_md5']!=md5:
            raise ValueError('Archive differs from pinned official source')
    a.archive_cache.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.archive_cache).free<3*sum(x['bytes'] for x in archives)+2*1024**3:
        raise ValueError('Insufficient space for retained archives, extraction and reserve')
    a.output.mkdir(parents=True);a.audio_output.mkdir(parents=True)
    (a.output/'INCOMPLETE').write_text('Audio staging incomplete.\n')
    (a.audio_output/'INCOMPLETE').write_text('Audio staging incomplete.\n')
    status=dict(schema=1,scope=__doc__,state='starting',pid=os.getpid(),active_download_pid=None,
                source='https://openslr.trmal.net/resources/12/',rate_limit=a.rate_limit,
                corpus_sha256=hashlib.sha256((a.corpus/'corpus.json').read_bytes()).hexdigest(),
                completed=[],training_started=False)
    def write():
        temporary=a.output/'status.pending'
        temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(a.output/'status.json')
    write()
    try:
        # Make real held-out speech available before the large training archive.
        for split in ('validation','test','train'):
            record=next(x for x in archives if x['split']==split)
            final=a.archive_cache/record['file'];part=final.with_suffix(final.suffix+'.part')
            status.update(state='download',active_archive=record['file'],expected_bytes=record['bytes'])
            write()
            if not final.exists():
                command=['curl','--fail','--location','--proto','=https','--proto-redir','=https',
                         '--silent','--show-error','--connect-timeout','20','--max-time','7200',
                         '--retry','3','--retry-delay','5','--speed-time','120','--speed-limit','1024',
                         '--limit-rate',a.rate_limit,'--continue-at','-','--output',str(part),status['source']+record['file']]
                with (a.output/(record['name']+'-download.log')).open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    status['active_download_pid']=child.pid
                    while child.poll() is None:
                        status['downloaded_bytes']=part.stat().st_size if part.exists() else 0
                        write();time.sleep(2)
                    status['active_download_pid']=None
                    if child.returncode:
                        raise ValueError('Download exited '+str(child.returncode)+'; partial retained')
                status.update(state='verify',downloaded_bytes=part.stat().st_size);write()
                verify(part,record);part.rename(final)
            else:
                status.update(state='verify');write();verify(final,record)
            status.update(state='extract');write()
            prep['extract_split'](final,record['name'],a.audio_output/record['name'])
            status['completed'].append(dict(split=split,file=record['file'],bytes=record['bytes'],sha256=record['sha256'],archive_verified=True,extracted=True))
            print('Verified and extracted '+record['file'],flush=True)
        status.update(state='complete',active_archive=None,active_download_pid=None)
        (a.output/'INCOMPLETE').unlink();(a.audio_output/'INCOMPLETE').unlink();write()
    except BaseException as error:
        status.update(state='failed',error=str(error),active_download_pid=None);write()
        raise


if __name__=='__main__':
    main()
