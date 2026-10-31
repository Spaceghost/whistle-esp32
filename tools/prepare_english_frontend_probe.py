#!/usr/bin/env python3
"""Make a frontend-only profile-3 carrier; never claim its weights were trained for profile 3."""
import argparse,hashlib,json,struct
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Preserve prior evidence; use a new output')
    report=json.loads((a.source/'export.json').read_text());original=(a.source/'english-student-research.cact').read_bytes()
    source_sha=hashlib.sha256(original).hexdigest()
    if source_sha!=report['sha256'] or report['audio_frontend_profile']!=2 or report['deployment_permitted'] is not False or (a.source/'INCOMPLETE').exists():raise ValueError('Need the complete retained profile-2 research export')
    h=struct.unpack_from('<48If',original)
    if h[0]!=0x31455741:raise ValueError('Need the fail-closed AWE1 research format')
    index=1+h[10]*27+9+2+h[31]*4+1
    rec=struct.unpack_from('<BBHIIIIQQII',original,196+4*h[2]+44*index)
    if rec[0:2]!=(2,1) or rec[8]!=52 or struct.unpack_from('<2f',original,rec[7])!=(3246.,2.):raise ValueError('Unexpected manifest')
    out=bytearray(original);offset=rec[7]+4;struct.pack_into('<f',out,offset,3.)
    assert out[:offset]==original[:offset] and out[offset+4:]==original[offset+4:]
    a.output.mkdir(parents=True);(a.output/'INCOMPLETE').write_text('Frontend fixture not complete.\n')
    (a.output/'english-student-research.cact').write_bytes(out)
    report.update(scope='Frontend-only fixture; neural weights were trained with profile 2',audio_frontend_profile=3,frontend_only_fixture=True,
                  weights_trained_with_frontend_profile=2,source_model_sha256=source_sha,sha256=hashlib.sha256(out).hexdigest(),
                  neural_payload_unchanged=True,tag_only_upgrade=False,native_validation_complete=False,deployment_permitted=False)
    (a.output/'export.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'INCOMPLETE').unlink()

if __name__=='__main__':main()
