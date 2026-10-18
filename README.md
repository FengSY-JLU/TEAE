# TEAE: Task-driven Enhancement Action Evolution

**Task-Beneficial Underwater Image Enhancement via Learned Action Manifolds and Policy Amortization**

**Authors:** Shiyao Feng, Jun-Hong Cui, ChunRu Wan, and Gaochao Xu  
**Paper status:** Manuscript prepared for submission to *IEEE Transactions on Image Processing* (TIP). The public manuscript, bibliographic details, and official BibTeX will be linked here when available.  
**Project status:** Frozen inference and benchmark evaluation code available; pretrained-model redistribution review in progress.

## Overview

![TEAE overview: privileged discovery, policy amortization, and deployment](assets/teae_overview.png)

*Fig. 1. Overview from the TEAE manuscript. Stage I uses privileged downstream task feedback to learn and search a source-dependent enhancement action repertoire; Stage II amortizes useful behavior into a single Student enhancer. **At deployment, only the frozen HRNet CorrectState provider and the C2-A Student run.** There is no Explorer, candidate-action search, test-time ground-truth feedback, or teacher-ensemble scoring.*

## Highlights

- **Learned action manifold:** An image-conditioned repertoire with 95 candidate slots, including a no-op action, is trained using privileged task feedback in Stage I.
- **Policy amortization:** Canonical-transition acquisition, basin-anchored refinement, and advantage-guided supervision lead to a deployable C2-A Student.
- **Frozen deployment:** A five-level HRNet CorrectState pyramid conditions a single Student forward pass; the privileged discovery and training pipeline is not needed at inference.
- **Task-centered evaluation:** The repository keeps frozen-network immediate utility (Protocol-A) distinct from independently retrained downstream models (Protocol-B).

## Evaluation snapshot (SUIM TEST110)

These metrics follow the frozen protocols provided in `benchmarks/` and are **specific to these task models and evaluation settings**; they do not represent a general image-quality ranking.

| Input | Protocol-A Mean-3T mIoU (%) | Protocol-B U-Net-VGG16 mIoU (%) | Protocol-B SegFormer-B2 mIoU (%) |
|---|---:|---:|---:|
| RAW | 73.615 | 66.267 | 72.808 |
| DTI-UIE | 67.835 | 62.922 | 73.269 |
| HSRUIE | 63.265 | 62.281 | 70.487 |
| **TEAE (C2-A)** | **75.596** | **69.289** | **75.746** |

**Protocol-A** uses three *frozen* segmentation networks and averages per-image GT-present-class hard mIoU (including class 0 when present, ignoring 255). It measures the immediate effect of enhancement without downstream retraining. TEAE gains **+1.981 percentage points** over RAW in Mean-3T mIoU.

**Protocol-B** retrains a separate U-Net-VGG16 or SegFormer-B2 for each of the four input domains, using their frozen E80 endpoints; its metric is dataset-global confusion-matrix mIoU. The two protocols have different model adaptation and aggregation semantics and **must not be mixed**.

**Validation status:** The released-source candidate was separately checked against the original model implementation. The final internal 110-source replay passed Protocol-A reference agreement and all eight Protocol-B endpoints using a predeclared bounded numerical tolerance (maximum absolute mIoU difference 0.01 percentage points); it was **not** a claim of bit-for-bit equal confusion matrices or of reproducing the original third-party authors' training protocols. See [`metadata/NUMERICAL_REPRODUCIBILITY.md`](metadata/NUMERICAL_REPRODUCIBILITY.md).

## Repository scope

| Included | Not included |
|---|---|
| C2-A Student and frozen HRNet CorrectState inference | Stage I Explorer / Oracle implementation and training checkpoints |
| Frozen Protocol-A teacher evaluation | Privileged candidate rollouts, action-manifold training code, training banks |
| Protocol-B evaluation for eight independent E80 segmenters | SSOR and unpublished follow-up research pipelines |
| Frozen benchmark evidence and fair DTI/HSR comparison protocol | Original DTI-UIE/HSRUIE code and pretrained weights; SUIM images and labels |

**This is an inference-and-assessment reproducibility release, not a complete end-to-end training reproduction.** Methodological details are described in the manuscript.

## Installation and model assets

A compatible PyTorch/CUDA environment is needed. The formal experiments used PyTorch `2.7.1+cu118`; `requirements.txt` lists Python packages but is not a fully frozen environment lockfile.

```bash
python -m pip install -r requirements.txt
```

When the model-release licensing review is complete, download the model archive from the [GitHub Releases page](https://github.com/FengSY-JLU/TEAE/releases) and extract it in the repository root:

```bash
tar -xzf TEAE_MODEL_RELEASE_20261018_151552.tar.gz
# This creates weights/ (12 model-only checkpoints and required configurations).
```

The model asset manifest is in [`metadata/release_models.json`](metadata/release_models.json). The prepared archive's SHA256 is
`993468f5fad3249014708bbde0855c2f5559ed90f67f3a6a50754b02c4cff6ab`.

**Availability note:** The archive may initially exist only as a draft Release in a private repository; a draft or private model asset is not a public download. Do not infer redistribution permission from the fact that checkpoint training was performed in our project.

## Enhancement inference

```bash
python -m inference.run \
  --weights-root weights \
  --student weights/teae_c2a_student.pt \
  --input /path/to/input_rgb_images \
  --output outputs/enhanced \
  --input-mode fixed256
```

Inputs and outputs are RGB images. The public inference path uses the frozen CorrectState provider plus one C2-A Student forward pass; segmentation labels are not used at inference.

## Reproduce Protocol-A / Protocol-B

After obtaining the official SUIM dataset, generating the baseline RGB images from their *upstream official inference*, and preparing TEAE outputs:

```bash
python -m evaluation.protocol_a --suim /path/to/SUIM --weights-root weights \
  --condition raw --condition dti=/path/to/dti_rgb \
  --condition hsr=/path/to/hsr_rgb --condition teae=/path/to/teae_rgb \
  --outdir outputs/protocol_a

python -m evaluation.protocol_b --suim /path/to/SUIM --weights-root weights \
  --checkpoint weights/protocol_b_segformer_b2_teae_e80.pth \
  --architecture segformer_b2 --domain teae --image-dir /path/to/teae_rgb \
  --output outputs/protocol_b/segformer_b2__teae.json

CUDA_VISIBLE_DEVICES=1 python -m scripts.reproduce_all --suim /path/to/SUIM \
  --weights-root weights --dti /path/to/dti_rgb --hsr /path/to/hsr_rgb \
  --teae /path/to/teae_rgb --outdir outputs/full_reproduction
```

Protocol-B's historical validation transform uses size 384, and its frozen scoring batches are 8 (U-Net) and 4 (SegFormer). DTI official256 output has a separate mask-geometry preparation step. See [`benchmarks/DTI_HSR_FAIR_EVALUATION_PROTOCOL.md`](benchmarks/DTI_HSR_FAIR_EVALUATION_PROTOCOL.md). All commands require users to supply their own data paths and correctly generated enhanced images.

## Third-party baselines and permissions

- **[DTI-UIE](https://github.com/oucailab/DTIUIE):** Use `underwater_test.py` and obtain the original pretrained checkpoint from the authors' official download instructions. The upstream repository specifies CC BY-NC 4.0 for non-commercial use.
- **[HSRUIE](https://github.com/gemyumeng/HSRUIE):** Use the official `test.py` and the checkpoint instructions in the authors' README. No original HSR weights are rehosted here.

The included code depends on third-party software and pretraining sources. In particular, NVIDIA's [SegFormer source license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) has non-commercial-use and redistribution/notice conditions; provenance and applicable terms for derived segmentation weights must be established before publishing model binaries. Please read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**No repository-wide open-source license is granted by this README.** Do not assume that all pretrained assets can be commercially used or redistributed.

## Paper and citation

**Task-Beneficial Underwater Image Enhancement via Learned Action Manifolds and Policy Amortization**  
Shiyao Feng, Jun-Hong Cui, ChunRu Wan, Gaochao Xu.

The manuscript is being prepared for *IEEE Transactions on Image Processing*. A paper URL, any preprint identifier, DOI and the official BibTeX entry will be added when they are actually available. **Please do not cite this work as accepted or published by TIP before a formal decision.**

## Questions

Please use this repository's Issues section for questions about the released inference interface and evaluation protocols. Check `metadata/` for the frozen evaluation conditions and release details.
