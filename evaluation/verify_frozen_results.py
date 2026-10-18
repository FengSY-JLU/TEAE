#!/usr/bin/env python3
"""Compare independently produced Protocol-A/B results with frozen public receipt.

Never claim an exact reproduction on the basis of frozen receipts alone.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence',type=Path,default=Path('benchmarks/frozen_benchmark_evidence.json'))
    ap.add_argument('--protocol-a-summary',type=Path)
    ap.add_argument('--protocol-b-dir',type=Path)
    ap.add_argument('--tolerance',type=float,default=1e-4,
                    help='Protocol-B numerical tolerance in [0,1] mIoU scale, at most 1e-4')
    ap.add_argument('--protocol-a-tolerance',type=float,default=5e-7,
                    help='Protocol-A retains the strict frozen evaluation tolerance')
    args=ap.parse_args()
    if not (0 <= args.tolerance <= 1e-4):
        ap.error('Protocol-B tolerance must be in [0, 1e-4]')
    if not (0 <= args.protocol_a_tolerance <= 5e-7):
        ap.error('Protocol-A tolerance must be in [0, 5e-7]')
    if args.protocol_a_summary is None and args.protocol_b_dir is None:
        ap.error('Provide new run output(s) to compare, not just frozen reference')
    ref=json.loads(args.evidence.read_text())
    count=0
    if args.protocol_a_summary:
        scored=json.loads(args.protocol_a_summary.read_text())['means']
        frozen=ref['protocol_a']['mean3t_miou_fraction']
        missing = sorted(set(frozen) - set(scored))
        if missing:
            raise RuntimeError(f'Missing required Protocol-A conditions: {missing}')
        if int(json.loads(args.protocol_a_summary.read_text()).get('source_count', -1)) != 110:
            raise RuntimeError('Protocol-A source count is not 110')
        for k,target in frozen.items():
            measured=float(scored[k]['mean3T_miou'])
            if abs(measured-target)>args.protocol_a_tolerance:
                raise RuntimeError(f'Protocol-A {k} mismatch {measured} vs {target}')
            print('A_PARITY',k,measured,flush=True);count+=1
    if args.protocol_b_dir:
        b=ref['protocol_b']['models']
        for arch,by_domain in b.items():
            for domain,expected in by_domain.items():
                path=args.protocol_b_dir/f'{arch}__{domain}.json'
                if not path.is_file():raise RuntimeError(f'Missing public Protocol-B run: {path}')
                measured=json.loads(path.read_text())
                if measured['source_count']!=110 or measured['architecture']!=arch or measured['domain']!=domain:
                    raise RuntimeError(f'Mismatched Protocol-B run scope: {path}')
                target=expected['dataset_global_mean_iou']
                if abs(float(measured['mean_iou'])-target)>args.tolerance:
                    raise RuntimeError(f'Protocol-B mismatch: {arch}/{domain}')
                print('B_PARITY',arch,domain,measured['mean_iou'],flush=True);count+=1
    if count==0:raise RuntimeError('Nothing was compared to independent fresh output')
    print('PASS_NEW_RUN_VS_FROZEN_REFERENCE',count,flush=True)

if __name__=='__main__':main()
