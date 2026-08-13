# MVDAF

Manuscript-aligned reference implementation for **Multi-View Dynamic Adaptive Fusion for Parotid Gland Tumor Classification in Multi-Sequence MRI**.

This repository was reconstructed from the locked manuscript protocol after historical development branches were cleaned from the working project. It preserves the final stage boundaries, tensor interfaces, leakage controls, and reported core hyperparameters, but it is not claimed to be a byte-for-byte archive of the original experiment scripts. Clinical data, trained checkpoints, patient-level predictions, and private fold manifests are not distributed.

MVDAF follows a leakage-controlled, three-stage protocol:

1. **Stage I — lesion-focused ROI extraction.** A dual-stream 2D/3D ResNet-50 model predicts a slice-level tumor score and uses box-supervised FPN attention to pool a 1024-dimensional ROI vector from the 2D `C4` feature map.
2. **Stage II — imbalance-aware representation learning.** Separate modality-specific ROI encoders are trained with class-balanced benign/malignant queues, supervised contrastive learning, and compactness regularization. The 128-dimensional projection is training-only.
3. **Stage III — patient-level fusion.** Frozen Stage II encoders transform **all** Stage I ROI vectors into 1024-dimensional class-aware slice features. Attention MIL aggregates each variable-length modality bag, and cross-attention fuses the three patient-level modality representations.

The code is organized under [`review/stage1`](review/stage1), [`review/stage2`](review/stage2), and [`review/stage3`](review/stage3). Historical development experiments, patient-level identifiers, prediction files, trained checkpoints, and clinical images are deliberately excluded.

## Installation

```bash
git clone git@github.com:xiwangderen/MVDAF.git
cd MVDAF
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data interface

The retrospective clinical MRI data cannot be redistributed in this repository. The expected de-identified manifests and feature files are described in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md). Keep all local data and fold splits outside version control.

## Reproduction order

Run each outer fold independently. Within an outer-training fold, create a fixed inner-training/inner-validation split, and then execute:

```bash
# Stage I: train one model per modality using inner-training patients only.
python -m review.stage1.train --config /path/to/stage1_fold_config.yaml

# Stage I: export ROI vectors and slice scores with the fold-specific checkpoint.
python -m review.stage1.extract --config /path/to/stage1_export_config.yaml

# Stage II: train one encoder per modality using only score-selected inner-training slices.
python -m review.stage2.train --config /path/to/stage2_fold_config.yaml

# Stage III: freeze Stage II encoders; train attention MIL and cross-attention fusion.
python -m review.stage3.train --config /path/to/stage3_fold_config.yaml
```

The public configuration templates encode the reported manuscript protocol and safe reference defaults. Paths, fold membership, and pretrained checkpoint locations must be supplied locally. See [`docs/FINAL_PROTOCOL.md`](docs/FINAL_PROTOCOL.md) for the separation of training-only candidate selection from final all-slice inference.

## Tests

The tests use synthetic tensors and contain no clinical data:

```bash
python -m compileall -q review
pytest -q
```

## Reproducibility and privacy scope

- Cross-validation splits are patient-level.
- Stage II queues are reinitialized per outer fold and populated only by inner-training patients.
- Checkpoints and thresholds are selected only on the inner-validation split.
- Outer-test patients are evaluated once and never enter queues or model selection.
- No patient data, identifiers, DICOM metadata, trained weights, or per-patient predictions are included.

## Citation

The arXiv identifier will be added after the authors complete the arXiv submission. Until then, please cite the manuscript title and authors listed in [`CITATION.cff`](CITATION.cff).

## License

Code is released under the [Apache License 2.0](LICENSE). The license does not grant access to, or rights over, the underlying clinical data.

