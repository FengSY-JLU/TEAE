# Third-party dependencies, checkpoints, and licenses

This repository contains TEAE research-authored inference and evaluation code. The status of each upstream component and checkpoint must be considered separately from the authorship of the project's training runs. The [public v0.1.0 Release](https://github.com/FengSY-JLU/TEAE/releases/tag/v0.1.0) distributes 12 project-trained model-only checkpoints for **non-commercial academic research and evaluation**, together with identified upstream license texts and provenance information. Publication of these files does **not** grant additional rights over third-party components.

## External enhancement baselines (not mirrored)

- **DTI-UIE:** https://github.com/oucailab/DTIUIE — the upstream repository identifies a Creative Commons Attribution-NonCommercial 4.0 International restriction and provides checkpoint download instructions. Readers obtain code and models from the upstream authors.
- **HSRUIE:** https://github.com/gemyumeng/HSRUIE — original source and checkpoint downloads remain with the upstream authors; redistribution authorization is not inferred from public availability.

## Architecture and pretrained-weight origins

- **SegFormer / MiT-B2:** Hugging Face `nvidia/mit-b2` identifies its license as `other` and points to NVIDIA's [SegFormer Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE). Where applicable, that license requires preservation of notices and a complete license copy, and restricts covered works and derivatives to non-commercial research/evaluation. TEAE documents the identified initialization lineage in [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md) and bundles the applicable full license in the model archive. A permissive license for surrounding framework code does not override restrictions on covered pretrained parameters.
- **PyTorch / torchvision / timm / Transformers and other dependencies:** These are maintained by their respective upstream authors and governed by their own licenses. Software-source terms and pretrained-weight terms should be considered separately; installation from a package index does not itself grant blanket rights to redistribute third-party weights.
- **VGG / HRNet backbone initializations:** Identified initialization families and provenance evidence are documented in [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md). Model weights, source code, and underlying training data may be governed by distinct terms; the release does not confer additional third-party rights.
- **Frozen evaluation-time transformation code:** The public implementation corresponds to the documented evaluation protocol. Any applicable upstream copyright and attribution obligations continue to apply; no blanket repository-wide license is implied.

## Published v0.1.0 model assets

The release includes a 12-checkpoint model archive with SHA256:

`2a2bbb2152b5e86d7f5275d83292bffc00f675a47851597f9e2f46be57c458ce`

The uploaded archive is named `TEAE_MODEL_RELEASE_v0.1.0_LICENSED_CANDIDATE.tar.gz`. The accompanying `SHA256SUMS.txt` uses the shorter name `TEAE_MODEL_RELEASE_v0.1.0.tar.gz`; users may rename the downloaded archive to match the checksum entry before verification. Renaming does not change file contents or SHA256.

The archive includes seven upstream license texts under `LICENSES/`: NVIDIA SegFormer, HRNet semantic segmentation, HRNet classification, NAFNet, TorchVision, timm, and Transformers. The source code and the pretrained weights do not necessarily carry identical permissions. Original DTI-UIE and HSRUIE enhancement-model checkpoints and dataset files are **not** included.

## Rights and use limitations

The published assets are intended for **non-commercial research/evaluation** subject to applicable upstream conditions. In particular, NVIDIA's SegFormer license (§§3.1–3.3) imposes notice/license-preservation requirements and non-commercial research/evaluation restrictions on covered works and derivatives. Checkpoints inheriting covered MiT-B2 parameters must be treated accordingly. TEAE does not grant commercial rights in third-party material.

**No repository-wide license is automatically granted by this document.** This notice summarizes identified obligations, does not certify rights in every upstream artifact, and does not replace the full license texts or any additional permissions that may be required.
