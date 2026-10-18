#!/usr/bin/env python3
"""Exact Protocol-B E80 evaluation, matching the historical TEST110 geometry.

Unlike Protocol-A, this protocol uses each domain's own geometry and the
historical validation segmentation transform (size 384) and E80 architecture-specific
batch sizes (U-Net 8, SegFormer 4). For DTI official256,
the GT is first resized to 256x256 by nearest-neighbor; then the common
validation transform is applied. We calculate dataset-global 8-class IoU.

The frozen historical transform source is vendored *from the exact original
segmentation snapshot* by internal/finalize_release_on_server.py. Its source
hash is stored in metadata/PROTOCOL_B_TRANSFORM_PROVENANCE.json. The public
scorer deliberately refuses to run without this vendored file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data.segmentation.loader import build_suim_loader
from teachers.frozen_ensemble import build_architecture

try:
    from evaluation.historical_validation_transform import ValidationSegmentationTransform
except ImportError as exc:
    raise RuntimeError(
        "Missing the exact historical ValidationSegmentationTransform; "
        "run internal/finalize_release_on_server.py to vendor the frozen "
        "transforms.py before release. Do not substitute an approximate resize."
    ) from exc

NCLASS = 8
DOMAINS = ('raw', 'dti', 'hsr', 'teae')
# Frozen in tools/baselines/train_protocolB_downstream_v1.py CANONICAL_EXPECTED.
# The historical TEST110 evaluator uses the same sizes as the E80 training.
FROZEN_E80_BATCH_SIZES = {'unet_vgg16': 8, 'segformer_b2': 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def get_source_id(batch):
    value = batch.get('source_id',batch.get('id',batch.get('name')))
    if isinstance(value, (list,tuple)):
        value=value[0]
    if torch.is_tensor(value):
        value=value.item()
    if value is None:
        raise ValueError('Missing test source id')
    return str(value)


def read_rgb_exact(path: Path):
    with Image.open(path) as im:
        a=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
    return torch.from_numpy(a).permute(2,0,1).float().div_(255.0).contiguous()


def nearest_mask_2d(mask: torch.Tensor, height: int, width: int):
    if mask.ndim != 2:
        raise ValueError('Expected unbatched HxW mask')
    return (F.interpolate(mask[None,None].float(),size=(height,width),
                          mode='nearest')[0,0].to(dtype=torch.long).contiguous())


def prepare_sample(batch, *, domain: str, image_dir: Path | None,
                   transform: ValidationSegmentationTransform):
    """Return the exact preprocessed CHW image / HW target for Protocol-B."""
    if domain not in DOMAINS:
        raise ValueError(domain)
    sid=get_source_id(batch)
    raw=batch['image']
    gt=batch['mask']
    if raw.ndim!=4 or gt.ndim!=3 or raw.shape[0]!=1 or gt.shape[0]!=1:
        raise ValueError('Expected unbatched SUIM loader with batch size 1')
    raw=raw[0].float().contiguous()
    mask=gt[0].long().contiguous()
    if raw.shape[-2:]!=mask.shape[-2:]:
        raise RuntimeError(f'{sid}: raw/mask geometry mismatch')
    if domain=='raw':
        image=raw.clone().contiguous()
    else:
        if image_dir is None:
            raise RuntimeError('Missing image-dir for '+domain)
        path=image_dir/(sid+'.png')
        if not path.is_file():
            raise FileNotFoundError(path)
        image=read_rgb_exact(path)
        if domain=='dti':
            if image.shape[-2:]!=(256,256):
                raise RuntimeError(f'{sid}: DTI official256 must be exactly 256x256')
            mask=nearest_mask_2d(mask,256,256)
        elif image.shape[-2:]!=raw.shape[-2:]:
            raise RuntimeError(f'{sid}: {domain} native source-geometry mismatch')
    if image.shape[-2:]!=mask.shape[-2:]:
        raise RuntimeError(f'{sid}: image/mask geometry mismatch before transform')
    valid=((mask>=0)&(mask<NCLASS))|(mask==255)
    if not bool(valid.all()):
        raise RuntimeError(f'{sid}: illegal mask labels')
    image,mask=transform(image,mask)
    if image.ndim!=3 or mask.ndim!=2 or image.shape[-2:]!=mask.shape[-2:]:
        raise RuntimeError(f'{sid}: historical transform malformed outputs')
    return sid,image.contiguous(),mask.long().contiguous()


def global_miou(matrix):
    mat=np.asarray(matrix,dtype=np.int64)
    if mat.shape!=(8,8):
        raise ValueError('Expected 8x8 matrix')
    tp=np.diag(mat)
    denom=mat.sum(0)+mat.sum(1)-tp
    valid=denom>0
    if not np.any(valid):
        raise ValueError('No evaluable classes')
    return float(np.mean(tp[valid]/denom[valid]))


def forward_batch_confusion(model, image_batch, mask_batch, device):
    """Frozen Protocol-B: one 8- or 4-image AMP forward, then pooled counts.

    No per-image prediction and no model-specific re-normalization. Final
    short batch is processed together, exactly as drop_last=False in the
    historical original DataLoader.
    """
    if len(image_batch)!=len(mask_batch) or not image_batch:
        raise ValueError('Empty/misaligned Protocol-B batch')
    image_shapes={tuple(x.shape) for x in image_batch}
    mask_shapes={tuple(y.shape) for y in mask_batch}
    if len(image_shapes)!=1 or len(mask_shapes)!=1:
        raise RuntimeError('Protocol-B historical transformation dimensions drift')
    x=torch.stack(image_batch,dim=0).to(device=device,dtype=torch.float32)
    y=torch.stack(mask_batch,dim=0).to(device=device,dtype=torch.long)
    with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
        output=model(x)
        if not isinstance(output,dict) or 'logits' not in output:
            raise RuntimeError('Unexpected segmentation network output')
        logits=output['logits']
        if logits.shape[-2:]!=y.shape[-2:]:
            logits=F.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
    pred=logits.argmax(dim=1)
    valid=(y>=0)&(y<NCLASS)&(y!=255)
    ids=(y[valid].to(torch.int64)*NCLASS+pred[valid].to(torch.int64))
    return torch.bincount(ids,minlength=64).reshape(8,8).cpu().numpy().astype(np.int64,copy=False)


def evaluate(args):
    if args.domain!='raw' and args.image_dir is None:
        raise ValueError('Non-raw domain requires --image-dir')
    if args.architecture not in FROZEN_E80_BATCH_SIZES:
        raise RuntimeError('No frozen Protocol-B batch contract for '+str(args.architecture))
    frozen_batch_size=FROZEN_E80_BATCH_SIZES[args.architecture]
    device=torch.device(args.device)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(ck,dict) or 'model_state_dict' not in ck or int(ck.get('epoch',-1))!=80:
        raise RuntimeError('Expected a model-only E80 checkpoint')
    model=build_architecture(args.architecture,num_classes=8,
        backbone_asset_root=args.weights_root/'assets/backbones/seg_teachers')
    incompatible=model.load_state_dict(ck['model_state_dict'],strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError('Strict E80 weight loading failed')
    model.to(device).eval().requires_grad_(False)
    loader=build_suim_loader(root=args.suim,split='test',batch_size=1,workers=0,
        shuffle=False,collate_mode='bank',pin_memory=False,persistent_workers=False)
    if len(loader.dataset)!=110:
        raise RuntimeError('Expected SUIM TEST110 with 110 sources')
    transform=ValidationSegmentationTransform(size=384,ignore_index=255)
    matrix=np.zeros((8,8),dtype=np.int64)
    seen=set()
    images=[]
    masks=[]
    completed=0
    next_progress=10
    with torch.inference_mode():
        for batch in loader:
            sid,image,mask=prepare_sample(batch,domain=args.domain,image_dir=args.image_dir,
                                           transform=transform)
            if sid in seen:
                raise RuntimeError('Duplicate SUIM source '+sid)
            seen.add(sid)
            images.append(image)
            masks.append(mask)
            if len(images)==frozen_batch_size:
                matrix+=forward_batch_confusion(model,images,masks,device)
                completed+=len(images)
                images.clear();masks.clear()
                if completed>=next_progress:
                    print('PROGRESS',completed,'/110',flush=True)
                    next_progress=((completed//10)+1)*10
        if images:
            matrix+=forward_batch_confusion(model,images,masks,device)
            completed+=len(images)
            images.clear();masks.clear()
            print('PROGRESS',completed,'/110',flush=True)
    if len(seen)!=110 or completed!=110:
        raise RuntimeError('Incomplete TEST110')
    score=global_miou(matrix)
    result={
        'protocol':'B','architecture':args.architecture,'domain':args.domain,
        'epoch':80,'source_count':110,'mean_iou':score,
        'metric':'global 8-class confusion matrix IoU, ignore255',
        'evaluation_transform':'historical ValidationSegmentationTransform(size=384,ignore_index=255)',
        'dti_geometry':'mask nearest->256 before historical validation transform',
        'amp':device.type=='cuda',
        'historical_batch_size':frozen_batch_size,
        'checkpoint_model_only_sha256':sha256(args.checkpoint),
        'confusion_matrix':matrix.tolist(),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS_PROTOCOL_B_EXACT_GEOMETRY_BATCH_SCORING',args.architecture,args.domain,
          f'{score:.12f}',f'batch={frozen_batch_size}',flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suim',type=Path,required=True)
    p.add_argument('--weights-root',type=Path,default=Path('weights'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--architecture',choices=['unet_vgg16','segformer_b2'],required=True)
    p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--image-dir',type=Path)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    evaluate(args)

if __name__=='__main__':
    main()
