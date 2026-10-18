# Third-party dependencies, checkpoints, and licenses

This repository contains TEAE research-authored inference and evaluation code. The status of *every* upstream component and checkpoint must be considered separately from the authorship of our training runs. **The complete 12-checkpoint model archive has not been cleared for public redistribution by this notice.**

## External enhancement baselines (not mirrored)

- **DTI-UIE:** https://github.com/oucailab/DTIUIE — upstream repository identifies a Creative Commons Attribution-NonCommercial 4.0 International restriction and provides checkpoint download instructions. Readers obtain code and models from the upstream authors.
- **HSRUIE:** https://github.com/gemyumeng/HSRUIE — original source and checkpoint downloads remain at the official repository; redistribution authorization is not inferred from public availability.

## Architecture and pretrained-weight origins

- **SegFormer / MiT-B2:** Hugging Face `nvidia/mit-b2` identifies its license as `other` and points to NVIDIA's [SegFormer Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE). That license requires, where applicable, redistribution under its terms, inclusion of the complete license and retention of notices; it also limits covered works and their derivatives to non-commercial research/evaluation use. The distinction between architecture reimplementation, pretrained parameter provenance, and fine-tuned checkpoint derivation must be recorded before redistributing TEAE's SegFormer-based checkpoint files. If a released binary is covered, include the required license and notices in its distribution, not merely this hyperlink.
- **PyTorch / torchvision / timm / Transformers and other dependencies:** distributed by their respective maintainers under their own licenses; verify library and pretrained weight sources independently. Running `pip install` does not grant rights to rehost pretrained files.
- **VGG / HRNet backbone initializations:** training in TEAE does not alone establish redistribution rights; verify the exact starting weight sources and their applicable terms before releasing model-only checkpoints.
- **Frozen evaluation-time transformation code:** included as an exact implementation copy from the project's evaluation snapshots. Confirm contributor authorship and obligations, if it contains any upstream-derived source, before assigning it a license.

## Release decision

1. Confirm intellectual-property rights for research-authored source, the figure, and every model checkpoint.
2. Map each checkpoint to its exact initialization origin (including upstream model card/version) and determine whether it incorporates covered source weights.
3. Bundle full upstream license texts and any required attribution notices with relevant distributed assets. Consider different terms for project-owned code versus third-party pretrained weights.
4. Keep the 12-checkpoint Release as a draft/private asset until the applicable conditions have been satisfied. A public source repository does not automatically license its binaries.

**No repository-wide license is automatically granted by this document.** This notice is informational and not a substitute for reviewing the applicable license texts or obtaining permission where needed.
