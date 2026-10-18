#!/usr/bin/env python3
"""Run public frozen SUIM Protocol-A V2 and the eight Protocol-B E80 endpoints.

This creates fresh measurements; the frozen evidence JSON is comparison-only.
Command runs sequentially to avoid loading all downstream models at once.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

B_ARCHS=('unet_vgg16','segformer_b2')
B_DOMAINS=('raw','dti','hsr','teae')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suim',type=Path,required=True)
    p.add_argument('--weights-root',type=Path,default=Path('weights'))
    p.add_argument('--dti',type=Path,required=True,help='Official256 RGB outputs, PNG named by SUIM source ID')
    p.add_argument('--hsr',type=Path,required=True,help='Official HSR native RGB outputs')
    p.add_argument('--teae',type=Path,required=True,help='Frozen C2-A RGB outputs')
    p.add_argument('--outdir',type=Path,default=Path('outputs/full_reproduction'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--tolerance',type=float,default=1e-4)
    args=p.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    dirs={'dti':args.dti,'hsr':args.hsr,'teae':args.teae}
    cmd=[sys.executable,'-m','evaluation.protocol_a','--suim',str(args.suim),
        '--weights-root',str(args.weights_root),'--device',args.device,
        '--condition','raw', '--outdir',str(args.outdir/'protocol_a')]
    for name,folder in dirs.items():cmd+=['--condition',name+'='+str(folder)]
    print('RUN_PROTOCOL_A',flush=True);subprocess.run(cmd,check=True)
    for arch in B_ARCHS:
        for domain in B_DOMAINS:
            ckpt=args.weights_root/f'protocol_b_{arch}_{domain}_e80.pth'
            cmd=[sys.executable,'-m','evaluation.protocol_b','--suim',str(args.suim),
                 '--weights-root',str(args.weights_root),'--checkpoint',str(ckpt),
                 '--architecture',arch,'--domain',domain,'--device',args.device,
                 '--output',str(args.outdir/'protocol_b'/f'{arch}__{domain}.json')]
            if domain!='raw':cmd+=['--image-dir',str(dirs[domain])]
            print('RUN_PROTOCOL_B',arch,domain,flush=True)
            subprocess.run(cmd,check=True)
    subprocess.run([sys.executable,'-m','evaluation.verify_frozen_results',
        '--protocol-a-summary',str(args.outdir/'protocol_a/summary.json'),
        '--protocol-b-dir',str(args.outdir/'protocol_b'),
        '--tolerance',str(args.tolerance)],check=True)
    print('PASS_TEAE_PUBLIC_FRESH_110_SOURCE_REPRODUCTION',flush=True)

if __name__=='__main__':main()
