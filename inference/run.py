#!/usr/bin/env python3
"""TEAE inference using the frozen five-level HRNet state and C2-A student.

The official evaluation input preprocessing should be selected explicitly.  For
convenient image inference, native-pad32 and fixed256 modes are provided.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.enhancer.nafnet_raw_anchored_task_state_v1 import RawAnchoredNAFNetTaskStateV1
from teachers.frozen_ensemble import load_frozen_teacher_ensemble
from models.teae.inference_output import resolve_enhanced_image

CHANNELS = (64, 128, 256, 512, 1024)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def get_five_level_state(teachers, image):
    result = teachers.forward_teacher('hrnet_w18', image)
    pyramid = result['feature_pyramid']
    if isinstance(pyramid, (tuple, list)):
        values = list(pyramid[:5])
    elif isinstance(pyramid, dict):
        values = [pyramid[k] for k in ('L0','L1','L2','L3','L4')]
    else:
        raise TypeError('Unexpected feature_pyramid representation')
    if len(values) != 5 or any(x.shape[1] != CHANNELS[i] for i, x in enumerate(values)):
        raise RuntimeError('Expected five CorrectState levels with channels 64,128,256,512,1024')
    return values


def load_student(path: Path, device, expected_sha: str | None = None):
    if expected_sha and sha256(path) != expected_sha:
        raise RuntimeError('Student model hash does not match the selected frozen release')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'student_state_dict' not in checkpoint:
        raise RuntimeError('Expected a model-only checkpoint with student_state_dict')
    model = RawAnchoredNAFNetTaskStateV1(
        width=32, encoder_blocks=(2,2,4,8), middle_blocks=12,
        decoder_blocks=(2,2,2,2), gate_floor=0.05, gate_ceiling=1.0,
        task_state_channels=CHANNELS, state_norm_groups=32,
    ).to(device)
    model.load_state_dict(checkpoint['student_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2,0,1).float().div(255).unsqueeze(0)


def make_input(original: Image.Image, mode: str):
    t = image_to_tensor(original)
    if mode == 'fixed256':
        # This is a configurable public path; see the benchmark contract for
        # precise per-method enhancement-output generation settings.
        t = F.interpolate(t, size=(256,256), mode='bilinear', align_corners=False)
        return t, (0,0)
    if mode == 'native-pad32':
        h,w = t.shape[-2:]
        dh = (-h) % 32
        dw = (-w) % 32
        if dh or dw:
            t = F.pad(t, (0,dw,0,dh), mode='replicate')
        return t, (dh,dw)
    raise ValueError(mode)


def render(model, teachers, raw: torch.Tensor):
    with torch.inference_mode():
        state = get_five_level_state(teachers, raw)
        enhanced = resolve_enhanced_image(model(raw, task_state_pyramid=state))
    if tuple(enhanced.shape) != tuple(raw.shape):
        raise RuntimeError('Unexpected enhanced tensor geometry')
    return enhanced.clamp(0,1)


def write_png(t: torch.Tensor, path: Path):
    image = t.squeeze(0).detach().float().cpu().permute(1,2,0).numpy()
    u8 = np.rint(np.clip(image,0,1)*255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8,'RGB').save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-root',type=Path,default=Path('weights'))
    p.add_argument('--student',type=Path,default=Path('weights/teae_c2a_student.pt'))
    p.add_argument('--input',type=Path,required=True,help='RGB image or folder')
    p.add_argument('--output',type=Path,required=True,help='PNG output folder')
    p.add_argument('--input-mode',choices=['fixed256','native-pad32'],default='fixed256')
    p.add_argument('--restore-original-size',action='store_true')
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--student-sha',default='',help='Optional release artifact SHA256')
    args=p.parse_args()
    device=torch.device(args.device)
    root=args.weights_root.resolve()
    teachers=load_frozen_teacher_ensemble(project_root=root,device=device,
                                          teacher_names=('hrnet_w18',),num_classes=8)
    model=load_student(args.student,device,args.student_sha or None)
    imgs=[args.input] if args.input.is_file() else sorted(x for x in args.input.iterdir()
              if x.is_file() and x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})
    if not imgs: raise SystemExit('No images found')
    for i,path in enumerate(imgs,1):
        with Image.open(path) as img:
            original=img.convert('RGB')
        raw,padding=make_input(original,args.input_mode)
        pred=render(model,teachers,raw.to(device))
        if args.input_mode=='native-pad32':
            dh,dw=padding
            if dh: pred=pred[...,:-dh,:]
            if dw: pred=pred[...,:,:-dw]
        if args.restore_original_size and pred.shape[-2:] != (original.height,original.width):
            pred=F.interpolate(pred,size=(original.height,original.width),mode='bilinear',align_corners=False)
        output=args.output/(path.stem+'.png')
        write_png(pred,output)
        print(f'[{i}/{len(imgs)}] {path.name} -> {output}',flush=True)
    print('PASS_TEAE_PUBLIC_INFERENCE',len(imgs),flush=True)

if __name__=='__main__':main()
