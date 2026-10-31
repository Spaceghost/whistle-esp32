#!/usr/bin/env python3
"""Compare complete legacy decoder output on cached natural YouTube speech."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

LEGACY_SHA='8c956a951107b03a1832a3c6d665c88e52cc59fadcf91397077416f22f5dca9b'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('before','after','model','catalog','audio-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output; prior evidence must be preserved')
    if sha(a.model)!=LEGACY_SHA:raise ValueError('Wrong original English model')
    catalog=json.loads(a.catalog.read_text());cases=catalog['cases']
    if catalog['kind']!='natural_youtube_speech' or len(cases)!=4 or len({c['video_id'] for c in cases})!=4:
        raise ValueError('Require the four independently sourced natural speech fixtures')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    root=a.audio_root.resolve();a.output.mkdir(parents=True);(a.output/'INCOMPLETE').write_text('Comparison not complete.\n')
    records=[]
    for i,c in enumerate(cases):
        path=(root/c['file']).resolve()
        if path.parent!=root or sha(path)!=c['sha256']:raise ValueError('Changed or escaped fixture')
        with wave.open(str(path),'rb') as f:
            if (f.getnchannels(),f.getsampwidth(),f.getframerate(),f.getnframes(),f.getcomptype())!=(1,2,16000,c['frames'],'NONE'):
                raise ValueError('Wrong natural speech geometry')
            pcm=array.array('h',f.readframes(f.getnframes()))
        if sys.byteorder!='little':pcm.byteswap()
        floats=array.array('f',(x/32768. for x in pcm));audio=a.output/f'{i}.f32';audio.write_bytes(floats.tobytes())
        result={}
        for name,binary in [('before',a.before),('after',a.after)]:
            stem=a.output/f'{i}-{name}'
            turn=subprocess.run([str(binary.resolve()),str(a.model.resolve()),'decode-pcm',str(audio.resolve()),str(stem.resolve())],capture_output=True,text=True,timeout=180)
            (a.output/f'{i}-{name}.stderr').write_text(turn.stderr)
            if turn.returncode:raise RuntimeError('Native decode failed: '+name+' '+c['id'])
            result[name]=json.loads(turn.stdout)
        b,n=result['before'],result['after']
        entry={'id':c['id'],'fixture_sha256':c['sha256'],'prefix':n['prefix'],
               'token_ids_identical':b['tokens']==n['tokens'],'unbiased_score_hex_identical':b['score_hex']==n['score_hex'],
               'encoder_bit_identical':sha(a.output/f'{i}-before.enc')==sha(a.output/f'{i}-after.enc'),
               'all_prefix_logits_bit_identical':sha(a.output/f'{i}-before.logits')==sha(a.output/f'{i}-after.logits'),
               'prefix_positions':len(n['tokens'])+2,'vocabulary_rows_per_position':n['vocabulary']}
        records.append(entry);print(json.dumps(entry),flush=True)
        if not all(entry[k] for k in ('token_ids_identical','unbiased_score_hex_identical','encoder_bit_identical','all_prefix_logits_bit_identical')):
            raise ValueError('Legacy regression; preserve outputs and investigate')
    report={'schema':1,'model_sha256':LEGACY_SHA,'baseline_binary_sha256':sha(a.before),'candidate_binary_sha256':sha(a.after),
            'cases':records,'all_cases_identical':True,'scope':'Host regression on four cached natural clips; not a microphone or new-model accuracy claim',
            'fixture_references_human_reviewed':catalog['reference_review']['human_reviewed'],'deployment_performed':False}
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'INCOMPLETE').unlink()


if __name__=='__main__':main()
