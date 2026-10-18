# TEAE: Task-driven Enhancement Action Evolution

**Task-Beneficial Underwater Image Enhancement via Learned Action Manifolds and Policy Amortization**

This repository provides the **frozen deployable TEAE enhancer**, its HRNet CorrectState dependency, frozen Protocol-A segmentation teachers, the eight independent Protocol-B downstream models, and scripts for running enhancement and evaluating the publicly documented benchmark. It does **not** provide a full training reproduction; the manuscript describes the training methodology.

## What is reproduced?

| Component | Public materials | Evaluation semantics |
|---|---|---|
| Deployment | HRNet-W18 CorrectState (five levels) + one frozen Student | Image-to-image inference; no GT at inference |
| Protocol-A V2 | Three frozen teachers; official RGB domain outputs; score script | Per-image GT-present-class hard-prediction mIoU, averaged per teacher and then over three teachers |
| Protocol-B | 8 independently domain-adapted E80 U-Net/SegFormer models | Dataset-global 8-class confusion-matrix mIoU |
| Baselines | Links to DTI-UIE/HSRUIE official inference, transparent geometry and scoring documentation | Not a reproduction of their original paper training protocols |

### Model downloads

Retrieve the 12 downloadable model artifacts from the project's GitHub **Releases** section when available. The official `metadata/release_models.json` lists the filenames and SHA256 checksums. **A Release should not be announced before those files and the clean-environment reproduction checks exist.** Downloaded weights should be extracted under `weights/` (ignored by Git). The `weights/assets/teachers/manifest.json` is part of the release's loader-compatible metadata.

```bash
python -m pip install -r requirements.txt
python -m inference.run --weights-root weights --student weights/teae_c2a_student.pt \
  --input examples/input --output outputs/enhanced --input-mode fixed256

python -m evaluation.protocol_a --suim /path/to/SUIM --weights-root weights \
  --condition raw --condition dti=/path/to/dti_rgb \
  --condition hsr=/path/to/hsr_rgb --condition teae=/path/to/teae_rgb \
  --outdir outputs/protocol_a

python -m evaluation.protocol_b --suim /path/to/SUIM --weights-root weights \
  --checkpoint weights/protocol_b_segformer_b2_teae_e80.pth \
  --architecture segformer_b2 --domain teae --image-dir /path/to/teae_rgb \
  --output outputs/protocol_b/segformer_b2__teae.json
```

To run all nine independent evaluations (1 Protocol-A + 8 Protocol-B), once all 12 model-only weights and the three RGB method directories are available:

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.reproduce_all --suim /path/to/SUIM \
 --weights-root weights --dti /path/to/dti_rgb --hsr /path/to/hsr_rgb \
 --teae /path/to/teae_rgb --outdir outputs/full_reproduction
```

This script rejects differences exceeding the declared mIoU tolerance (default 1e-4 in [0,1] scale, or 0.01 percentage point). A fresh replay has **not** been completed until this command succeeds in an isolated clone.

Protocol-A and Protocol-B metrics **cannot be combined directly**. Their distinct scoring definitions and frozen results are documented in `benchmarks/`.

### Reproducibility limitations

The published code supports enhancement inference and frozen-model assessment; it does not reproduce the entire model training procedure from scratch. The original third-party baseline implementations, their weight files, and SUIM data are not mirrored here. The output generation protocol and the precise official/native spatial configurations are material to fair comparison. For a paper-exact reproduction, validate the RGB generation pipeline and frozen-model evaluation on the 110 official SUIM sources.

### Usage and attribution

Check `THIRD_PARTY_NOTICES.md` for third-party model/backbone and baseline licensing. See the manuscript for scientific definitions. Any release should be dated, checksummed, and validated in a clean checkout.

## Protocol-B numerical reproducibility

The formal Protocol-B evaluator uses the frozen 384-pixel validation transform, CUDA AMP, batch size 8 for U-Net-VGG16 and batch size 4 for SegFormer-B2, and one pooled 8x8 confusion matrix over TEST110. For DTI official256, the GT mask is nearest-neighbor resized to 256 before the 384 validation transform. Small backend/AMP rounding differences are possible even with identical weights and input preprocessing. We therefore report *bounded numerical agreement*, not bit-for-bit reproduction. The internal release gate requires mIoU absolute difference <=1e-4 (fraction, 0.01 percentage point), exact GT-class row sums and normalized aggregate confusion L1/(2N) <=0.0005 (this is not a pixelwise disagreement rate). Report actual measured deltas alongside frozen scores. The thresholds are predeclared and may not be widened automatically. Confusion matrix aggregation cannot establish per-pixel equality. See `benchmarks/` for published reference results.

The complete action-manifold learning and privileged teacher/Oracle training implementation is not included. Refer to the manuscript for the training methodology. Third-party model and training-transform redistribution rights must be checked before publication.
