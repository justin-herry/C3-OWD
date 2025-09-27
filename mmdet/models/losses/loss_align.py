# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from ..builder import LOSSES
from .utils import weighted_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def alignment_loss(projected_visual, text_embeds):
    """Compute alignment loss between projected visual features and text embeddings.

    Args:
        projected_visual (torch.Tensor): Normalized projected visual features
            with shape (N, D).
        text_embeds (torch.Tensor): Normalized text embeddings with shape (N, D).

    Returns:
        torch.Tensor: Computed alignment loss per sample.
    """
    # Compute similarity matrix
    logits = projected_visual @ text_embeds.t()
    
    # Create target labels (diagonal elements are matches)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Calculate cross entropy loss per sample
    loss = F.cross_entropy(logits, labels, reduction='none')
    return loss


@LOSSES.register_module()
class AlignmentLoss(nn.Module):
    """Alignment loss between projected visual features and text embeddings.

    Args:
        reduction (str, optional): Reduction method for the loss.
            Options are 'none', 'mean', and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AlignmentLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    @staticmethod
    def _get_src_permutation_idx(indices):
        """Get source permutation indices from matching results.

        Args:
            indices (list[tuple]): Matching results between predictions and targets.

        Returns:
            tuple: Batch indices and source indices.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self,
                outputs,
                targets,
                indices,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function for alignment loss.

        Args:
            outputs (dict): Model outputs containing:
                - visual_features: Raw visual features
                - proj_visual_features: Projected visual features
                - text_embeddings: Text embeddings
            targets (list[dict]): Target annotations.
            indices (list[tuple]): Matching indices between predictions and targets.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor for loss reduction.
            reduction_override (str, optional): Override reduction method.

        Returns:
            torch.Tensor: Calculated alignment loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        # Extract indices for positive samples
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        
        # Get projected visual features for matched indices
        projected_visual = outputs["proj_visual_features"][batch_idx, src_idx]
        projected_visual = projected_visual / projected_visual.norm(dim=1, keepdim=True)

        # Get corresponding text embeddings
        target_classes = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        text_embeds = outputs["text_embeddings"][target_classes]
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # Calculate alignment loss
        loss_align = alignment_loss(
            projected_visual,
            text_embeds,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor
        )

        return self.loss_weight * loss_align