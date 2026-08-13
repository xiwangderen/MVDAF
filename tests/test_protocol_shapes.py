import torch

from review.stage1.fpn import FPNSpatialAttention
from review.stage1.model import normalized_attention_pool
from review.stage2.losses import DualQueueSupConCompactness
from review.stage2.model import ModalityROIEncoder, ProjectionHead
from review.stage3.model import MVDAFPatientClassifier


def test_stage1_attention_and_roi_shapes():
    head = FPNSpatialAttention(fpn_dim=16).eval()
    with torch.no_grad():
        logits = head(
            torch.randn(2, 512, 28, 28),
            torch.randn(2, 1024, 14, 14),
            torch.randn(2, 2048, 7, 7),
        )
        roi = normalized_attention_pool(
            torch.randn(2, 1024, 14, 14), torch.sigmoid(logits)
        )
    assert logits.shape == (2, 1, 28, 28)
    assert roi.shape == (2, 1024)


def test_stage2_projection_is_training_only_128d():
    encoder = ModalityROIEncoder().eval()
    projector = ProjectionHead().eval()
    with torch.no_grad():
        encoded = encoder(torch.randn(8, 1024))
        projected = projector(encoded)
    assert encoded.shape == (8, 1024)
    assert projected.shape == (8, 128)
    assert torch.allclose(projected.norm(dim=1), torch.ones(8), atol=1e-5)


def test_dual_queue_warmup_and_loss():
    objective = DualQueueSupConCompactness(capacity=16, queue_samples_per_class=4)
    labels = torch.tensor([0, 0, 1, 1])
    first, _ = objective(torch.randn(4, 128), labels, compact_weight=0.0)
    second, logs = objective(torch.randn(4, 128), labels, compact_weight=0.16)
    assert first.ndim == 0 and second.ndim == 0
    assert torch.isfinite(second)
    assert set(logs) == {"supcon", "compact"}


def test_stage3_uses_all_variable_length_slices():
    model = MVDAFPatientClassifier().eval()
    for encoder in model.encoders.values():
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    lengths = {m: [3, 2] for m in ("T1A", "T2A", "T2C")}
    bags = {m: torch.randn(5, 1024) for m in ("T1A", "T2A", "T2C")}
    with torch.no_grad():
        output = model(bags, lengths)
    assert output["logits"].shape == (2,)
    assert output["modality_weights"].shape == (2, 3)
    assert torch.allclose(output["modality_weights"].sum(dim=1), torch.ones(2), atol=1e-5)
    for modality in lengths:
        assert [len(weights) for weights in output["slice_weights"][modality]] == [3, 2]

