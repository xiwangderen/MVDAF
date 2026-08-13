# Final manuscript protocol

This document records the protocol implemented by the public code and prevents historical development branches from being mistaken for the final method.

## Stage I

- Input: a center slice and its local three-slice window, resized to `224 x 224`.
- 2D backbone outputs `C3 (512 x 28 x 28)`, `C4 (1024 x 14 x 14)`, and `C5 (2048 x 7 x 7)` under the standard ResNet/FPN naming convention.
- The 3D branch supplies local inter-slice context.
- The FPN attention head predicts a `28 x 28` spatial response supervised by Gaussian maps derived from bounding boxes; negative slices use zero maps.
- The main ROI vector is obtained by resizing attention to `14 x 14` and performing normalized attention pooling on `C4`, yielding 1024 dimensions.
- Stage I assigns a tumor score to every slice and does not itself discard slices.

## Stage II

- Candidate construction retains slices with `p_L >= 0.75`; if no slice meets the threshold for a patient/modality, the highest-scoring slice is retained by a deterministic label-free fallback.
- T1A, T2A, and T2C use independent 1024-dimensional ROI encoders.
- A training-only projection head maps encoded features to 128-dimensional, L2-normalized embeddings.
- Separate benign and malignant FIFO queues are maintained per modality, with capacity 4096 per class.
- Queue samples are balanced across the two classes before entering the contrastive set.
- Queues are gradient-detached, reinitialized for every outer fold, and populated only from inner-training patients.
- The Stage II objective is supervised contrastive loss plus within-class compactness regularization (weight 0.16, enabled from epoch 4; malignant compactness weight 1.25).
- After training, projection heads are discarded. Only the frozen 1024-dimensional encoder outputs proceed to Stage III.

## Stage III

- All available slices are retained; the Stage II candidate threshold is not reused for final classification.
- Each modality forms a variable-length bag of frozen 1024-dimensional class-aware slice features.
- Attention MIL generates a modality-level patient representation and slice-reliance weights.
- A learned query performs cross-attention over the T1A, T2A, and T2C patient representations.
- The classifier is optimized with focal loss (`alpha=0.75`, `gamma=2.0`).

## Leakage control

For every outer fold, model fitting and Stage II queue construction use only inner-training patients. Inner-validation patients are used only for checkpoint and threshold selection. Outer-test patients are excluded from all training, queue, checkpoint, and threshold operations.

## Release scope

This repository is a clean reference implementation reconstructed from the locked manuscript protocol. It is not a byte-for-byte archive of the historical experiment scripts, which are no longer present in the working project. Reproducing the numerical tables additionally requires the private patient-level folds, MRI data, preprocessing provenance, pretrained weights, and locked checkpoints, none of which are distributed here.
