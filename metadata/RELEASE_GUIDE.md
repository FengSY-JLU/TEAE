# Model Release plan

The release builder produces a `weights-release/` folder with 12 sanitized, model-only files and compatible teacher manifests. These artifacts are kept **outside the Git source ZIP**, and must be uploaded as a GitHub Release after confirming redistribution rights. Model SHA256 after reserialization differs from the original training checkpoint; keep both digests in `release_models.json`.

The 8 Protocol-B models are independent domain-trained E80 endpoints; they are not interchangeable with the three Protocol-A teachers. Keep test dataset files external; no SUIM images/masks are included.

Do not commit large model files to Git history. Model loading scripts accept SHA256 checks for downloaded release assets. Before public release, run the fresh 110-source Protocol-A/B evaluation and compare against `benchmarks/frozen_benchmark_evidence.json`. The published model source alone is not a verified benchmark reproduction.
