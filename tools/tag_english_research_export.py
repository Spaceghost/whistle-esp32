#!/usr/bin/env python3
"""Upgrade the retained pre-release profile-2 export to the fail-closed AWE1 tag.

Only the container tag changes. This neither trains nor changes any weight,
configuration, tokenizer or feature rule. The original is always preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Choose a fresh output directory')
    report=json.loads((a.source/'export.json').read_text())
    original=(a.source/'english-student-research.cact').read_bytes()
    before=hashlib.sha256(original).hexdigest()
    if (a.source/'INCOMPLETE').exists() or before!=report['sha256'] or report['audio_frontend_profile']!=2 or report['deployment_permitted'] is not False:
        raise ValueError('Source must be a complete, verified, non-deployable profile-2 research export')
    cfg=report['config']
    if cfg['languages']!=['en'] or cfg['text']!=report['text_vocabulary'] or report['embedding_rows']!=cfg['text']+1:
        raise ValueError('Source is not the English research configuration')
    h=struct.unpack_from('<48If',original)
    if h[0]!=0x05E12A84 or h[5]!=cfg['text']+1:raise ValueError('Unexpected preliminary container')
    index=1+h[10]*27+9+2+h[31]*4+1
    rec=struct.unpack_from('<BBHIIIIQQII',original,196+4*h[2]+44*index)
    if rec[0:2]!=(2,1) or rec[8]!=52 or struct.unpack_from('<3f',original,rec[7])!=(3246.,2.,float(cfg['text'])):
        raise ValueError('Unexpected preliminary frontend manifest')
    tagged=bytearray(original);struct.pack_into('<I',tagged,0,0x31455741)
    assert tagged[4:]==original[4:]
    a.output.mkdir(parents=True);(a.output/'INCOMPLETE').write_text('Research format upgrade not complete.\n')
    path=a.output/'english-student-research.cact'
    with path.open('xb') as target:target.write(tagged);target.flush();os.fsync(target.fileno())
    report.update(native_format='AWE1',sha256=hashlib.sha256(tagged).hexdigest(),
                  parent_export_sha256=before,tag_only_upgrade=True,payload_sha256=hashlib.sha256(original[4:]).hexdigest(),
                  format_upgrade_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'export.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'INCOMPLETE').unlink()
    print(json.dumps({'bytes':len(tagged),'sha256':report['sha256'],'native_format':'AWE1','payload_unchanged':True}),flush=True)


if __name__=='__main__':main()
