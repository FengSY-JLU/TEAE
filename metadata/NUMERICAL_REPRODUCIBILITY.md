# Protocol-B numerical reproduction standard

The official E80 SUIM TEST110 paper result is retained without modification.

Public evaluation uses the original validation transform (384 pixels), AMP on CUDA, batch 8 (U-Net) or 4 (SegFormer), dataset-global 8x8 confusion-matrix mIoU and 110 matched sources.

A fresh run is considered numerically consistent only when: (i) the exact GT-class support histogram from each row of the confusion matrix matches, (ii) the absolute mIoU difference from the frozen result is at most **1e-4** on the [0,1] scale (**0.01 percentage point**), (iii) the normalized aggregate confusion-matrix L1 difference divided by twice the total number of evaluated pixels is at most **0.0005**, and (iv) all eight endpoint model/data identities are correct. Such a pass **does not claim exact integer confusion-matrix equality**.

Original scores are never replaced with rounded replay scores. Internal run reports preserve per-endpoint mIoU difference and confusion L1 difference.
