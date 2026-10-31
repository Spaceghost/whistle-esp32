#!/usr/bin/env python3
"""Exact row-format size arithmetic, not a quantizer or model acceptance test."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Preserve earlier estimates; choose a new output')
    data=a.model.read_bytes();h=struct.unpack_from('<48If',data)
    if h[0]!=0x31455741:raise ValueError('Expected AWE1 English research artifact')
    start=196+4*h[2]
    records=[struct.unpack_from('<BBHIIIIQQII',data,start+i*44) for i in range(h[1])]
    audio=1+h[10]*27+9+2+h[31]*4+1
    config=struct.unpack_from('<13f',data,records[audio][7])
    enc_start=audio+1+h[10]*10+1
    recipes=[('current-four-bit',lambda i:4),('all-cq-two-bit',lambda i:2),
             ('head-four-rest-two',lambda i:4 if i==0 else 2),
             ('head-and-decoder-four-rest-two',lambda i:4 if i<1+h[10]*27+9 else 2),
             ('head-four-encoder-three-rest-two',lambda i:4 if i==0 else 3 if enc_start<=i<h[1]-1 else 2)]
    results=[]
    for name,choose in recipes:
        position=start+h[1]*44;matrix_counts={}
        for i,r in enumerate(records):
            position=(position+63)//64*64;size=r[8]
            if r[0]==3:
                rows,columns,group=r[3],r[4],r[9]
                padded=(columns+group-1)//group*group
                if size!=rows*(padded*r[10]//8+padded//group*2):raise ValueError('Unexpected CQ row format')
                bits=choose(i);size=rows*(padded*bits//8+padded//group*2)
                matrix_counts[str(bits)]=matrix_counts.get(str(bits),0)+1
            position+=size
        results.append({'policy':name,'estimated_bytes':position,'estimated_saved_vs_research_bytes':len(data)-position,
                        'fits_current_watch_model_partition_by_size_only':position<=0xBE0000,'cq_matrix_counts_by_bits':matrix_counts})
    if results[0]['estimated_bytes']!=len(data):raise ValueError('Size arithmetic does not reproduce actual artifact')
    report={'schema':1,'source_model_sha256':hashlib.sha256(data).hexdigest(),'actual_four_bit_bytes':len(data),'watch_model_partition_bytes':0xBE0000,
            'geometry':{'text_vocabulary':int(config[2]),'encoder_layers':int(config[3]),'decoder_layers':h[10],'dimension':h[7],'engram_sites':h[31]},
            'policies':results,'weights_changed':False,'deployment_permitted':False,
            'qualification':'Predicted sizes only. Every changed precision policy needs matching QAT, a real export, held-out speech and target acceptance. Topology and FP16/FP32/raw payload lengths are held fixed.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(results),flush=True)


if __name__=='__main__':main()
