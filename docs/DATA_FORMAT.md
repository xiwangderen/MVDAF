# De-identified data interfaces

Clinical images are not distributed. The training scripts expect user-created, de-identified manifests whose subject keys are study-local pseudonyms rather than medical record numbers.

## Stage I image manifest

CSV columns:

| Column | Meaning |
|---|---|
| `subject_key` | De-identified patient key |
| `modality` | `T1A`, `T2A`, or `T2C` |
| `slice_index` | Index within the volume |
| `slice_path` | Path to a float32 `.npy` center slice shaped `[224,224]` |
| `prev_path` | Previous slice, or center slice at the boundary |
| `next_path` | Next slice, or center slice at the boundary |
| `slice_label` | `1` when tumor is visible, otherwise `0` |
| `x1,y1,x2,y2` | Bounding box in resized pixel coordinates; empty for negative slices |
| `patient_label` | `0` benign, `1` malignant |

## Stage II/III feature files

Each de-identified patient/modality `.npz` file contains:

- `roi`: float32 array `[num_slices, 1024]`.
- `p_l`: float32 array `[num_slices]` containing Stage I tumor scores.
- `label`: scalar integer (`0` benign, `1` malignant).
- `slice_index`: optional integer array `[num_slices]`.

A feature manifest contains `subject_key`, `modality`, and `feature_path`. Fold membership is supplied through separate local files and is never inferred from file names.

## Required privacy checks

Before creating these files, remove direct identifiers and DICOM metadata. Do not commit manifests, local paths, folds, features, model outputs, or checkpoints to Git.

