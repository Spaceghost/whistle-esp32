#!/usr/bin/env python3
"""Compile old/new host decoders and compare synthetic speech and decode heap.

Uses the caller's C compiler and scratch directory. Never runs on a device,
changes model files or downloads data. Baseline source is read from Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import struct
import wave

WRAPPER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
void *__real_malloc(size_t);
void *__real_calloc(size_t,size_t);
void __real_free(void *);
static struct {void *pointer; size_t bytes;} tracked[4096];
static size_t live,peak,logits_bytes,logits_calls,allocations;
static int measuring;
void memory_begin(size_t bytes) { logits_bytes=bytes; measuring=1; }
static void record(void *p,size_t bytes) {
    if(!measuring || !p)return;
    for(unsigned i=0;i<4096;i++)if(!tracked[i].pointer) {
        tracked[i].pointer=p;tracked[i].bytes=bytes;live+=bytes;
        if(live>peak)peak=live;
        if(bytes==logits_bytes)logits_calls++;
        allocations++;return;
    }
    abort();
}
void *__wrap_malloc(size_t bytes) {void *p=__real_malloc(bytes);record(p,bytes);return p;}
void *__wrap_calloc(size_t n,size_t bytes) {void *p=__real_calloc(n,bytes);record(p,n*bytes);return p;}
void __wrap_free(void *p) {
    if(p)for(unsigned i=0;i<4096;i++)if(tracked[i].pointer==p) {
        live-=tracked[i].bytes;tracked[i].pointer=NULL;break;
    }
    __real_free(p);
}
void memory_end(void) {
    measuring=0;
    fprintf(stderr,"decode_heap peak=%zu live=%zu logits_calls=%zu allocations=%zu\n",peak,live,logits_calls,allocations);
}
"""

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('wav',type=Path)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    model,wav=args.model.resolve(),args.wav.resolve()
    if args.report.exists():
        parser.error('Report already exists; choose a new evidence path')
    with wave.open(str(wav),'rb') as audio:
        if (audio.getnchannels(),audio.getframerate(),audio.getsampwidth(),audio.getcomptype())!=(1,16000,2,'NONE'):
            parser.error('Fixture must be 16 kHz mono PCM16 WAV')
    header=model.read_bytes()[:24]
    if len(header)!=24 or struct.unpack_from('<I',header,20)[0]!=8193:
        parser.error('Expected the 8193-row English-conditioned Whistle artifact')
    baseline=subprocess.check_output(['git','rev-parse',args.baseline+'^{commit}'],cwd=root,text=True).strip()
    old=subprocess.check_output(['git','show',baseline+':engine/src/nd_whistle.c'],cwd=root,text=True)
    driver=(root/'host/wh_run.c').read_text()
    call='    rc = nd_whistle_decode(&w, enc, T, lang, ids, 320);'
    assert driver.count(call)==1
    driver=driver.replace(call,'    memory_begin(w.vocab*sizeof(float));\n'+call+'\n    memory_end();')
    driver='#include <stddef.h>\nvoid memory_begin(size_t);\nvoid memory_end(void);\n'+driver
    driver=driver.replace('score %.3f','score %a')
    sources=[str(p) for p in (root/'engine/src').glob('*.c') if p.name!='nd_whistle.c']
    report={'schema':1,'baseline_commit':baseline,'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),'host_only':True,
        'no_firmware_or_model_deployed':True,'monolingual_neural_pruning_proven':False,'cases':[]}
    with tempfile.TemporaryDirectory(prefix='whistle-decode-memory-') as scratch:
        task=Path(scratch)
        (task/'before.c').write_text(old);(task/'driver.c').write_text(driver);(task/'memory.c').write_text(WRAPPER)
        for name,source in [('before',task/'before.c'),('after',root/'engine/src/nd_whistle.c')]:
            subprocess.run([os.environ.get('CC','cc'),'-O2','-std=gnu99','-I'+str(root/'engine/include'),
                str(task/'driver.c'),str(task/'memory.c'),str(source),*sources,'-lm',
                '-Wl,--wrap=malloc','-Wl,--wrap=calloc','-Wl,--wrap=free','-o',str(task/name)],check=True)
        for name,extra in [('plain',[]),('keywords',['--keywords','Levi,Ackerman,Attack on Titan,app','--bonus','4'])]:
            results={}
            for version in ('before','after'):
                run=subprocess.run([str(task/version),str(model),str(wav),'--lang','en',*extra],capture_output=True,text=True,check=True,timeout=180)
                selected=[line for line in run.stderr.splitlines() if line.startswith(('tokens ','score '))]
                assert len(selected)==2
                metrics=next(line for line in run.stderr.splitlines() if line.startswith('decode_heap '))
                metrics={k:int(v) for k,v in (part.split('=') for part in metrics.split()[1:])}
                results[version]={'stdout':run.stdout,'selected':selected,'metrics':metrics}
            before,after=results['before'],results['after']
            assert before['stdout']==after['stdout'] and before['selected']==after['selected'],'Transcript, token IDs or exact score differs'
            assert before['metrics']['live']==after['metrics']['live']==0,'Decoder leaked tracked heap'
            assert before['metrics']['logits_calls']==2 and after['metrics']['logits_calls']==1
            saved=before['metrics']['peak']-after['metrics']['peak']
            assert saved==8193*4
            report['cases'].append({'name':name,'transcript_identical':True,'token_ids_identical':True,
                'score_float_bits_identical':True,'before':before['metrics'],'after':after['metrics'],
                'decode_peak_heap_bytes_saved':saved})
            print(name+': exact transcript/tokens/score; decode peak -'+str(saved)+' bytes',flush=True)
    with args.report.open('x') as out:out.write(json.dumps(report,indent=2)+'\n')
    print('PASS '+str(args.report))

if __name__=='__main__':main()
