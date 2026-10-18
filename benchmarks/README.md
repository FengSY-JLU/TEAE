# Reproducing DTI-UIE and HSRUIE comparisons

We distinguish **upstream image generation** from **common downstream scoring**. The scored inputs were produced by the authors' official inference implementations and saved before evaluating with our frozen segmentation models. We do not claim to reproduce the original methods' published, differently configured experiments.

- DTI-UIE upstream: <https://github.com/oucailab/DTIUIE>, official `underwater_test.py`, pinned generation revision `fb09f9143decfb06a3bb557b78e0b15acbd9dcf0`, official checkpoint SHA256 `e9449842c0f81211f72bceacba63a7d1d2c085c920e00901e26e2e40dea0c7b6`. Main comparison is *official256 generation, common geometry scoring*. Keep native evaluation as a **separate diagnostic**.
- HSRUIE upstream: <https://github.com/gemyumeng/HSRUIE>, official `test.py`, native generation path. Pin the exact upstream commit and weight digest in your local run receipt (do not infer them from this repository).

Examples of **upstream-provided** inference entrypoints (run these in their own environments after preparing datasets and checkpoints exactly as required by their authors):

```bash
# DTI-UIE (official README example; not yet a SUIM-adapted wrapper)
python underwater_test.py --model model_dtiuie --draw_images

# HSRUIE (official README example; --dataroot must contain its testA input folder)
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

For our SUIM comparison, input/output directories and any size normalization must follow the frozen per-method generation receipts, not be guessed from these illustrative README commands. DTI's fixed256 and HSR's native generation conditions must remain distinct. See the frozen evidence document for the matching domain IDs.

The official baseline code and pretrained models are **not vendored or redistributed**. Follow the originating author's licence. Record the external code revision, model SHA, command line, per-image input/output shapes, file hashes, and output folder. Do not use segmentation GT during enhancement image generation.

**Protocol-A V2** freezes the same three segmentation teachers for all methods, reloads RGB from disk, resizes enhancement outputs to the corresponding RAW/GT tensor geometry using bilinear `align_corners=False`, then computes per-source hard-prediction GT-present-class mIoU (including background 0 if present; ignoring 255). DTI native-eval diagnostics are not aggregated with DTI official256.

**Protocol-B** trains a separate U-Net and SegFormer for each input domain. The eight evaluated E80 model checkpoints are included in the release asset manifest. The metric is **dataset-global confusion-matrix mIoU**, not Protocol-A per-image mIoU.

`frozen_benchmark_evidence.json` is a cryptographically checked report from the original official evaluation artifacts, **not a new execution**. To substantiate independent reproduction, run the public evaluation scripts and compare their newly generated outputs against this receipt.

Protocol-B training implementation and intermediate image-domain preparation are not provided here; these E80 weights support inference and scoring reproducibility. A complete retraining reproduction would need the same frozen adaptation training contract and input-domain generation assets.
