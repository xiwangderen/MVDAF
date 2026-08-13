"""Class-balanced dual-queue supervised contrastive objective."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DualQueueSupConCompactness(nn.Module):
    """Per-modality benign/malignant queues with detached FIFO updates."""

    def __init__(
        self,
        dim: int = 128,
        capacity: int = 4096,
        temperature: float = 0.07,
        queue_samples_per_class: int | None = None,
        malignant_compact_weight: float = 1.25,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.temperature = temperature
        self.queue_samples_per_class = queue_samples_per_class
        self.malignant_compact_weight = malignant_compact_weight
        self.register_buffer("queues", torch.zeros(2, capacity, dim))
        self.register_buffer("pointers", torch.zeros(2, dtype=torch.long))
        self.register_buffer("sizes", torch.zeros(2, dtype=torch.long))

    @torch.no_grad()
    def reset(self) -> None:
        self.queues.zero_()
        self.pointers.zero_()
        self.sizes.zero_()

    @torch.no_grad()
    def _enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach(), dim=1)
        for class_index in (0, 1):
            values = embeddings[labels == class_index]
            for value in values:
                pointer = int(self.pointers[class_index])
                self.queues[class_index, pointer] = value
                self.pointers[class_index] = (pointer + 1) % self.capacity
                self.sizes[class_index] = min(int(self.sizes[class_index]) + 1, self.capacity)

    def _valid_queue(self, class_index: int) -> torch.Tensor:
        return self.queues[class_index, : int(self.sizes[class_index])]

    def _prototypes(self) -> torch.Tensor | None:
        if (self.sizes == 0).any():
            return None
        return F.normalize(
            torch.stack([self._valid_queue(0).mean(0), self._valid_queue(1).mean(0)]), dim=1
        )

    def _sample_queue(
        self,
        class_index: int,
        prototypes: torch.Tensor,
        balanced_count: int,
    ) -> torch.Tensor:
        queue = self._valid_queue(class_index)
        count = balanced_count
        if count == queue.shape[0]:
            return queue.detach()
        hard_count = count // 2
        opposite = prototypes[1 - class_index]
        hardness = queue @ opposite
        hard = torch.topk(hardness, k=hard_count, largest=True).indices
        remaining_mask = torch.ones(queue.shape[0], dtype=torch.bool, device=queue.device)
        remaining_mask[hard] = False
        remaining = torch.flatnonzero(remaining_mask)
        random_count = count - hard_count
        random = remaining[torch.randperm(remaining.numel(), device=queue.device)[:random_count]]
        return queue[torch.cat([hard, random])].detach()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        compact_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = F.normalize(embeddings, dim=1)
        labels = labels.long()
        prototypes = self._prototypes()
        if prototypes is None:
            self._enqueue(embeddings, labels)
            zero = embeddings.sum() * 0.0
            return zero, {"supcon": zero.detach(), "compact": zero.detach()}

        balanced_count = min(int(self.sizes[0]), int(self.sizes[1]))
        queue_b = self._sample_queue(0, prototypes, balanced_count)
        queue_m = self._sample_queue(1, prototypes, balanced_count)
        contrast = torch.cat([embeddings, queue_b, queue_m], dim=0)
        contrast_labels = torch.cat(
            [
                labels,
                torch.zeros(queue_b.shape[0], dtype=torch.long, device=labels.device),
                torch.ones(queue_m.shape[0], dtype=torch.long, device=labels.device),
            ]
        )
        logits = embeddings @ contrast.T / self.temperature
        self_mask = torch.zeros_like(logits, dtype=torch.bool)
        self_mask[:, : embeddings.shape[0]] = torch.eye(
            embeddings.shape[0], dtype=torch.bool, device=embeddings.device
        )
        positives = labels[:, None].eq(contrast_labels[None, :]) & ~self_mask
        denominator = ~self_mask
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        log_prob = logits - torch.logsumexp(logits.masked_fill(~denominator, float("-inf")), dim=1, keepdim=True)
        positive_count = positives.sum(dim=1).clamp_min(1)
        supcon = -((log_prob * positives).sum(dim=1) / positive_count).mean()

        prototype = prototypes[labels]
        class_weight = torch.where(
            labels == 1,
            torch.as_tensor(self.malignant_compact_weight, device=labels.device),
            torch.ones((), device=labels.device),
        )
        compact = (class_weight * (1.0 - (embeddings * prototype).sum(dim=1))).mean()
        total = supcon + float(compact_weight) * compact
        self._enqueue(embeddings, labels)
        return total, {"supcon": supcon.detach(), "compact": compact.detach()}

