#!/usr/bin/env python3
"""Real-container profile admission/refusal checks; no device or model deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('binary','model','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--legacy-binary',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    source=a.model.read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    if digest!='8c956a951107b03a1832a3c6d665c88e52cc59fadcf91397077416f22f5dca9b':raise ValueError('Wrong retained legacy artifact')
    h=struct.unpack_from('<48If',source)
    start=196+h[2]*4
    audio_index=1+h[10]*27+9+2+h[31]*4+1
    audio=struct.unpack_from('<BBHIIIIQQII',source,start+audio_index*44)[7]
    tokenizer=struct.unpack_from('<BBHIIIIQQII',source,start+(h[1]-1)*44)[7]
    marker=source.index(b'<|en|>',tokenizer)
    mutations=[('legacy-profile-1',None,None,True),('profile-2-format-only',audio+4,struct.pack('<f',2.),True),('profile-3-format-only',audio+4,struct.pack('<f',3.),True)]
    for value in (0.,4.,float('nan'),float('inf')):mutations.append(('unknown-profile-'+str(value),audio+4,struct.pack('<f',value),False))
    for label,value in [('fractional',8192.5),('out-of-vocabulary',8193.),('not-English-tag',2048.),('nan',float('nan'))]:
        mutations.append(('text-prefix-'+label,audio+8,struct.pack('<f',value),False))
    mutations += [('stock-tag-with-profile-2',audio+4,struct.pack('<f',2.),False),
                  ('wrong-manifest-magic',audio,struct.pack('<f',3245.),False),
                  ('wrong-English-marker',marker,b'<|xx|>',False),
                  ('wrong-tokenizer-bos',tokenizer+12,struct.pack('<I',0),False),
                  ('wrong-tokenizer-vocabulary',tokenizer,struct.pack('<I',8192),False),
                  ('wrong-embedding-rows',start+4,struct.pack('<I',2049),False),
                  ('zero-window',audio+44,struct.pack('<f',0.),False),
                  ('zero-hop',audio+48,struct.pack('<f',0.),False)]
    a.output.mkdir(parents=True);results=[];older_refused=None
    for i,(name,offset,payload,accepted) in enumerate(mutations):
        candidate=bytearray(source)
        if payload is not None:candidate[offset:offset+len(payload)]=payload
        if name in ('profile-2-format-only','profile-3-format-only'):struct.pack_into('<I',candidate,0,0x31455741)
        path=a.output/(str(i)+'.cact');path.write_bytes(candidate)
        turn=subprocess.run([str(a.binary.resolve()),str(path.resolve()),'inspect'],capture_output=True,text=True,timeout=30)
        actual=turn.returncode==0
        results.append(dict(name=name,expected_accepted=accepted,actual_accepted=actual,exit_code=turn.returncode))
        print(json.dumps(results[-1]),flush=True)
        if actual!=accepted:
            (a.output/'failed.json').write_text(json.dumps(results,indent=2)+'\n');raise ValueError('Unexpected admission result; failing candidate preserved')
        if name in ('profile-2-format-only','profile-3-format-only') and a.legacy_binary:
            old=subprocess.run([str(a.legacy_binary.resolve()),str(path.resolve()),'inspect'],capture_output=True,timeout=30)
            older_refused=old.returncode==4 and old.stderr.startswith(b'open -2')
            if not older_refused:raise ValueError('Older engine must refuse the research format')
        path.unlink()  # Verified disposable mutation, reproducible from retained source.
    report={'schema':1,'scope':'Container admission only; changing a legacy artifact to profile 2 or 3 does not make its weights suitable for that frontend',
            'source_sha256':digest,'native_binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'checks':results,'older_engine_refused_research_tag':older_refused,'all_passed':True,'deployment_performed':False}
    (a.output/'admission.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
