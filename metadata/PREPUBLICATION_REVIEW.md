# TEAE public source package — independent source review

- Source archive was compiled and inspected independently after assembly.
- The public Student, HRNet CorrectState path, three frozen Protocol-A segmentation teachers, and eight Protocol-B independent downstream endpoints are represented in this source tree.
- The deployment parity logs supplied by the authors reported zero elementwise difference for a deterministic 256x256 probe between the original and staged implementations. These logs are not a substitute for a fresh TEST110 reproduction using exported model-only weights.
- Protocol-A V2: RGB read from disk, bilinear image resize to corresponding RAW spatial geometry (`align_corners=False`), frozen three teachers, GT-present-class hard IoU, including GT-present background class 0, ignoring label 255, average over sources and teachers.
- Protocol-B: eight independently adapted E80 U-Net / SegFormer checkpoints and dataset-global 8x8 confusion-matrix IoU (not Protocol-A per-image mean IoU).
- The reference comparison now requires all four principal Protocol-A conditions (RAW/DTI/HSR/TEAE) and exactly 110 sources, so an incomplete fresh evaluation cannot be misreported as successful.
- Stage I learning and privileged training source code are not provided. This is an inference and frozen-evaluation reproducibility release, not an end-to-end training reproduction.
- The upstream baseline inference recipes must be run according to their original implementations and their true data/geometry constraints. Documentation here does not establish that the original baseline inference has been re-run in a new environment.
- Third-party licence review, clean-clone TEST110 replay, and public model asset hosting remain release prerequisites.
