# DTI-UIE and HSRUIE baseline reproduction (official inference + common evaluator)

TEAE's public benchmark should reproduce both the original image-generation step and the controlled downstream scoring step. These are distinct operations.

## 1. Upstream inference sources

- DTI-UIE official source: https://github.com/oucailab/DTIUIE (`underwater_test.py`). Its README describes a CC BY-NC 4.0 non-commercial licensing restriction. Obtain code and weights according to the authors' instructions.
- HSRUIE official source: https://github.com/gemyumeng/HSRUIE (`test.py`, README checkpoint instructions). Check applicable permission before re-distributing any code or weights.

Pin an upstream Git commit, checkpoint SHA256, environment, exact command, native input/output geometry, and output-image hashes in a separate run receipt. Never silently replace an upstream inference implementation or apply a different learned checkpoint.

## 2. Protocol-A V2 (frozen task utility)

Use the same 110 SUIM evaluation sources and the official baseline-generated RGB outputs. DTI-UIE used the `dtiuie_official256` official256 generation path for the main comparison; the separate `dtiuie_official256_native_eval` diagnostic is never merged into that result. HSRUIE used `hsruie_native`. Reload the saved RGB outputs and evaluate using the same frozen three segmentation teachers, identical preprocessing, and common image/mask evaluation geometry. Align logits to GT mask geometry with the frozen scorer; compute hard argmax and per-image GT-present-class IoU, including class 0 when GT-present, ignoring class 255. Average over the images and the three teachers.

**Important**: identical *scoring geometry* does not imply that different authors' enhancers originally operated at identical native resolution or used identical training data. Record original upstream generation and any crop/resize policy explicitly. Never use SUIM test labels during baseline enhancement inference.

## 3. Protocol-B (independently adapted downstream models)

For each input domain RAW, DTI-UIE, HSRUIE and TEAE, use an independently trained U-Net-VGG16 and SegFormer-B2 E80 model with the frozen canonical downstream training contract. The eight evaluated checkpoints are published separately with SHA256 and model identities. Report dataset-global confusion-matrix mIoU (not Protocol-A per-image mIoU). Record canonical init, train split, optimizer, schedule, and input-domain generation separately to make the adaptation reproducible.

## 4. What the evidence manifest proves

The machine-generated `frozen_benchmark_evidence.json` ties the frozen Protocol-A summary/CSV hashes to **10 distinct conditions, each with the same 110 sources** and maps eight Protocol-B E80 checkpoints to the official TEST110 evaluation JSON. It is **not** a substitute for actually executing each upstream enhancement method and re-scoring output images in a clean environment. The full public evaluation entrypoints must still be tested.

## 5. Reuse and scope

Our evaluation wrappers, schemas and public manifests may be released under the project's applicable license, but original upstream implementations and weights are not rehosted by default. This benchmark component contains no privileged model-training code or unpublished research assets.

## Protocol-B exact geometry clarification (October 2026)

The official frozen E80 scoring code uses `ValidationSegmentationTransform(size=384, ignore_index=255)` on all domains **before** inference. For DTI-UIE `official256`, the 256x256 RGB output is paired with a nearest-neighbor-resized semantic mask at 256x256 **before** the historical validation transform. The original validation runner enables CUDA AMP. The global 8x8 confusion matrix is accumulated across TEST110 and then class IoUs are averaged.

Do **not** score native-resolution logits against the native masks and call that Protocol-B: an earlier pre-release evaluator omitted the transform and failed frozen E80 result parity. Protocol-A V2 is a different, frozen-three-teacher per-image metric and must not reuse the Protocol-B transform.

The byte-exact frozen historical transform source and its SHA256 are included in the finalized repository only after the internal one-shot finalizer validates source-level provenance and preprocessing parity. Publication requires clearance of the underlying transform's redistribution rights.
