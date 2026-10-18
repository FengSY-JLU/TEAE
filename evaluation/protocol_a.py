#!/usr/bin/env python3
"""Protocol-A V2: common-geometry, frozen three-teacher SUIM assessment.

Inputs are previously generated RGB images, not the proprietary training process.
Ground truth is consulted only in evaluation, never in enhancement generation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data.segmentation.loader import build_suim_loader
from teachers.frozen_ensemble import load_frozen_teacher_ensemble
from evaluation.protocol_a_metrics import gt_present_metrics

TEACHERS=('unet_vgg16','segformer_b2','hrnet_w18')


def get_source_id(batch):
    sid=batch.get('source_id',batch.get('id',batch.get('name')))
    if isinstance(sid,(list,tuple)): sid=sid[0]
    if torch.is_tensor(sid): sid=sid.item()
    if sid is None: raise ValueError('Missing source_id in SUIM loader')
    return str(sid)


def load_png(path):
    with Image.open(path) as img:
        arr=np.asarray(img.convert('RGB'),dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2,0,1).float().div(255).unsqueeze(0)


def score_one(teachers,image,mask):
    result={}
    for name in TEACHERS:
        pred=teachers.forward_teacher(name,image)
        miou,dice=gt_present_metrics(pred['logits'],mask)
        result[name+'_miou']=float(miou)
        result[name+'_dice']=float(dice)
    result['mean3T_miou']=float(np.mean([result[n+'_miou'] for n in TEACHERS]))
    result['mean3T_dice']=float(np.mean([result[n+'_dice'] for n in TEACHERS]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suim',type=Path,required=True,help='SUIM dataset root containing test split')
    p.add_argument('--weights-root',type=Path,default=Path('weights'))
    p.add_argument('--condition',action='append',required=True,metavar='NAME=IMAGE_FOLDER',
                   help='Enter multiple conditions; raw is available without an image folder')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    condition_dirs={}
    for item in args.condition:
        name,sep,path=item.partition('=')
        if not name or (name!='raw' and (not sep or not path)) or name in condition_dirs:
            raise SystemExit('Malformed or repeated --condition; expected raw or dti=PATH')
        condition_dirs[name]=None if name=='raw' else Path(path).resolve()
    device=torch.device(args.device)
    loader=build_suim_loader(root=args.suim,split='test',batch_size=1,workers=0,
             shuffle=False,collate_mode='bank',pin_memory=False,persistent_workers=False)
    teachers=load_frozen_teacher_ensemble(project_root=args.weights_root,device=device,
                                           teacher_names=TEACHERS,num_classes=8)
    if len(loader.dataset)!=110: raise RuntimeError('Formal SUIM TEST110 count mismatch')
    seen=set(); rows=[]
    with torch.inference_mode():
        for batch in loader:
            sid=get_source_id(batch)
            if sid in seen: raise RuntimeError('Duplicated SUIM source '+sid)
            seen.add(sid)
            original=batch['image'].to(device=device,dtype=torch.float32)
            target=batch['mask'].to(device=device)
            if original.shape[-2:]!=target.shape[-2:]: raise RuntimeError('RAW/GT geometry mismatch')
            for name,folder in condition_dirs.items():
                if folder is None:
                    x=original
                    old_shape=tuple(x.shape[-2:])
                else:
                    path=folder/(sid+'.png')
                    if not path.is_file():raise RuntimeError('Missing enhanced RGB: '+str(path))
                    x=load_png(path)
                    old_shape=tuple(x.shape[-2:])
                    if x.shape[-2:]!=original.shape[-2:]:
                        x=F.interpolate(x,size=original.shape[-2:],mode='bilinear',align_corners=False)
                    x=x.to(device=device,dtype=torch.float32)
                score=score_one(teachers,x,target)
                rows.append({'source_id':sid,'condition':name,'original_height':old_shape[0],
                         'original_width':old_shape[1],**score})
            if len(seen)%10==0 or len(seen)==1:print('PROGRESS',len(seen),'/',len(loader.dataset),flush=True)
    if len(seen)!=110 or len(rows)!=110*len(condition_dirs):raise RuntimeError('Missing evaluation rows')
    means={name:{key:float(np.mean([r[key] for r in rows if r['condition']==name]))
               for key in ('mean3T_miou','mean3T_dice',*[n+'_miou' for n in TEACHERS])}
           for name in condition_dirs}
    args.outdir.mkdir(parents=True,exist_ok=True)
    with (args.outdir/'per_source_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    (args.outdir/'summary.json').write_text(json.dumps({'protocol':'A-V2',
      'source_count':110,'condition_count':len(condition_dirs),
      'metric':'per-image GT-present class IoU including class0 if present, ignore255',
      'averaging':'images per teacher, then frozen teachers',
      'means':means},indent=2)+'\n')
    print('PASS_PROTOCOL_A_PUBLIC_SCORING',json.dumps(means),flush=True)

if __name__=='__main__':main()
