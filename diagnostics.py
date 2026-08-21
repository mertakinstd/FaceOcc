"""Validation-only diagnostics for FaceOcc scientific experiments.

The legacy FaceOcc checkpoint-selection metric remains ``IoU(threshold=0.0)``
from ``loss.py``.  This module only adds diagnostic observables and must not be
used to select checkpoints unless a future experiment explicitly changes the
protocol.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


DEFAULT_PROBABILITY_THRESHOLDS: Tuple[float, ...] = tuple(
    round(value, 2) for value in torch.arange(0.10, 0.901, 0.05).tolist()
)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float) -> torch.Tensor:
    return numerator / denominator.clamp_min(eps)


def _binary_boundary(mask: torch.Tensor, dilation_ratio: float) -> torch.Tensor:
    """Convert a binary mask to the inner boundary band used by Boundary IoU.

    Cheng et al. define the boundary width as a fraction of the image diagonal.
    Repeated 3x3 erosion by ``d`` pixels is equivalent to one square erosion
    with radius ``d``.  Explicit zero padding ensures objects touching the image
    border are treated as having a boundary, matching the reference algorithm.
    """

    if not 0.0 < dilation_ratio < 1.0:
        raise ValueError('boundary dilation_ratio must be in (0, 1)')
    height, width = mask.shape[-2:]
    dilation = max(1, int(round(dilation_ratio * math.sqrt(height ** 2 + width ** 2))))
    padded = F.pad(mask, (dilation, dilation, dilation, dilation), mode='constant', value=0.0)
    kernel_size = 2 * dilation + 1
    eroded = -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1, padding=0)
    return (mask - eroded).clamp_(0.0, 1.0)


class ValidationDiagnostics:
    """Accumulate per-image segmentation diagnostics without changing training."""

    def __init__(
        self,
        *,
        probability_thresholds: Iterable[float] = DEFAULT_PROBABILITY_THRESHOLDS,
        boundary_dilation_ratio: float = 0.02,
        ece_bins: int = 15,
        eps: float = 1e-7,
    ) -> None:
        thresholds = tuple(float(value) for value in probability_thresholds)
        if not thresholds:
            raise ValueError('at least one probability threshold is required')
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError('probability thresholds must be in (0, 1)')
        if ece_bins < 2:
            raise ValueError('ece_bins must be >= 2')

        self.probability_thresholds = thresholds
        self.boundary_dilation_ratio = float(boundary_dilation_ratio)
        self.ece_bins = int(ece_bins)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._device = None
        self._sample_count = 0
        self._iou_values = []
        self._sums: Dict[str, torch.Tensor] = {}
        self._threshold_iou_sum = None
        self._ece_count = None
        self._ece_probability_sum = None
        self._ece_positive_sum = None
        self._threshold_tensor = None

    def _ensure_state(self, device: torch.device) -> None:
        if self._device is not None:
            return
        self._device = device
        dtype = torch.float64
        zero = lambda: torch.zeros((), device=device, dtype=dtype)
        for key in (
            'precision',
            'recall',
            'dice',
            'soft_iou',
            'soft_dice',
            'full_bce',
            'brier',
            'boundary_iou',
            'fp_pixels',
            'fn_pixels',
            'fp_rate',
            'fn_rate',
            'gt_face_fraction',
            'pred_face_fraction',
        ):
            self._sums[key] = zero()

        self._threshold_iou_sum = torch.zeros(
            len(self.probability_thresholds), device=device, dtype=dtype
        )
        self._ece_count = torch.zeros(self.ece_bins, device=device, dtype=dtype)
        self._ece_probability_sum = torch.zeros(self.ece_bins, device=device, dtype=dtype)
        self._ece_positive_sum = torch.zeros(self.ece_bins, device=device, dtype=dtype)
        self._threshold_tensor = torch.tensor(
            self.probability_thresholds, device=device, dtype=torch.float32
        ).view(-1, 1, 1, 1, 1)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 4 or targets.ndim != 4:
            raise ValueError('expected logits and targets with shape (N, C, H, W)')
        if logits.shape != targets.shape:
            raise ValueError(f'logits/targets shape mismatch: {logits.shape} vs {targets.shape}')
        if logits.shape[1] != 1:
            raise ValueError('FaceOcc diagnostics expect one binary logit channel')

        self._ensure_state(logits.device)
        eps = self.eps
        batch_size = int(logits.shape[0])
        pixels_per_image = int(logits.shape[1] * logits.shape[2] * logits.shape[3])

        probs = torch.sigmoid(logits)
        gt = (targets > 0.5).to(logits.dtype)
        pred = (logits > 0.0).to(logits.dtype)  # exact legacy decision boundary: p > 0.5

        reduce_dims = (1, 2, 3)
        tp = torch.sum(pred * gt, dim=reduce_dims, dtype=torch.float64)
        fp = torch.sum(pred * (1.0 - gt), dim=reduce_dims, dtype=torch.float64)
        fn = torch.sum((1.0 - pred) * gt, dim=reduce_dims, dtype=torch.float64)
        tn = torch.sum((1.0 - pred) * (1.0 - gt), dim=reduce_dims, dtype=torch.float64)

        iou = _safe_ratio(tp, tp + fp + fn, eps)
        precision = _safe_ratio(tp, tp + fp, eps)
        recall = _safe_ratio(tp, tp + fn, eps)
        dice = _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn, eps)

        soft_tp = torch.sum(probs * gt, dim=reduce_dims, dtype=torch.float64)
        soft_pred = torch.sum(probs, dim=reduce_dims, dtype=torch.float64)
        soft_gt = torch.sum(gt, dim=reduce_dims, dtype=torch.float64)
        soft_iou = _safe_ratio(soft_tp, soft_pred + soft_gt - soft_tp, eps)
        soft_dice = _safe_ratio(2.0 * soft_tp, soft_pred + soft_gt, eps)

        full_bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
        full_bce = full_bce.mean(dim=reduce_dims, dtype=torch.float64)
        brier = torch.square(probs - gt).mean(dim=reduce_dims, dtype=torch.float64)

        gt_boundary = _binary_boundary(gt, self.boundary_dilation_ratio)
        pred_boundary = _binary_boundary(pred, self.boundary_dilation_ratio)
        boundary_intersection = torch.sum(
            gt_boundary * pred_boundary, dim=reduce_dims, dtype=torch.float64
        )
        boundary_union = torch.sum(
            (gt_boundary + pred_boundary) > 0, dim=reduce_dims, dtype=torch.float64
        )
        boundary_iou = _safe_ratio(boundary_intersection, boundary_union, eps)

        self._iou_values.append(iou.detach())
        self._sample_count += batch_size
        self._sums['precision'] += precision.sum()
        self._sums['recall'] += recall.sum()
        self._sums['dice'] += dice.sum()
        self._sums['soft_iou'] += soft_iou.sum()
        self._sums['soft_dice'] += soft_dice.sum()
        self._sums['full_bce'] += full_bce.sum()
        self._sums['brier'] += brier.sum()
        self._sums['boundary_iou'] += boundary_iou.sum()
        self._sums['fp_pixels'] += fp.sum()
        self._sums['fn_pixels'] += fn.sum()
        self._sums['fp_rate'] += _safe_ratio(fp, fp + tn, eps).sum()
        self._sums['fn_rate'] += _safe_ratio(fn, tp + fn, eps).sum()
        self._sums['gt_face_fraction'] += (soft_gt / pixels_per_image).sum()
        self._sums['pred_face_fraction'] += ((tp + fp) / pixels_per_image).sum()

        threshold_pred = (probs.unsqueeze(0) > self._threshold_tensor.to(probs.dtype)).to(probs.dtype)
        gt_expanded = gt.unsqueeze(0)
        threshold_tp = torch.sum(
            threshold_pred * gt_expanded, dim=(2, 3, 4), dtype=torch.float64
        )
        threshold_fp = torch.sum(
            threshold_pred * (1.0 - gt_expanded), dim=(2, 3, 4), dtype=torch.float64
        )
        threshold_fn = torch.sum(
            (1.0 - threshold_pred) * gt_expanded, dim=(2, 3, 4), dtype=torch.float64
        )
        threshold_iou = _safe_ratio(
            threshold_tp,
            threshold_tp + threshold_fp + threshold_fn,
            eps,
        )
        self._threshold_iou_sum += threshold_iou.sum(dim=1)

        # Pixel-probability calibration ECE.  This measures how closely the
        # predicted face probability in each bin matches the observed face
        # frequency.  It is diagnostic only and is intentionally not a model-
        # selection metric.
        flat_probs = probs.reshape(-1)
        flat_gt = gt.reshape(-1)
        for bin_index in range(self.ece_bins):
            lower = bin_index / self.ece_bins
            upper = (bin_index + 1) / self.ece_bins
            if bin_index == self.ece_bins - 1:
                mask = (flat_probs >= lower) & (flat_probs <= upper)
            else:
                mask = (flat_probs >= lower) & (flat_probs < upper)
            count = mask.sum(dtype=torch.float64)
            self._ece_count[bin_index] += count
            self._ece_probability_sum[bin_index] += flat_probs[mask].sum(dtype=torch.float64)
            self._ece_positive_sum[bin_index] += flat_gt[mask].sum(dtype=torch.float64)

    def compute(self) -> Dict[str, float]:
        if self._sample_count == 0:
            return {}

        count = float(self._sample_count)
        iou_values = torch.cat(self._iou_values).to(torch.float64)
        quantiles = torch.quantile(
            iou_values,
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=iou_values.device, dtype=torch.float64),
        )

        logs: Dict[str, float] = {
            'diag_iou_mean': float(iou_values.mean().item()),
            'diag_iou_p10': float(quantiles[0].item()),
            'diag_iou_p25': float(quantiles[1].item()),
            'diag_iou_median': float(quantiles[2].item()),
            'diag_iou_p75': float(quantiles[3].item()),
            'diag_iou_p90': float(quantiles[4].item()),
            'diag_precision_mean': float((self._sums['precision'] / count).item()),
            'diag_recall_mean': float((self._sums['recall'] / count).item()),
            'diag_dice_mean': float((self._sums['dice'] / count).item()),
            'diag_soft_iou_mean': float((self._sums['soft_iou'] / count).item()),
            'diag_soft_dice_mean': float((self._sums['soft_dice'] / count).item()),
            'diag_full_bce_mean': float((self._sums['full_bce'] / count).item()),
            'diag_brier_mean': float((self._sums['brier'] / count).item()),
            'diag_boundary_iou_mean': float((self._sums['boundary_iou'] / count).item()),
            'diag_fp_pixels_mean': float((self._sums['fp_pixels'] / count).item()),
            'diag_fn_pixels_mean': float((self._sums['fn_pixels'] / count).item()),
            'diag_fp_rate_mean': float((self._sums['fp_rate'] / count).item()),
            'diag_fn_rate_mean': float((self._sums['fn_rate'] / count).item()),
            'diag_gt_face_fraction_mean': float((self._sums['gt_face_fraction'] / count).item()),
            'diag_pred_face_fraction_mean': float((self._sums['pred_face_fraction'] / count).item()),
        }

        nonempty = self._ece_count > 0
        total_pixels = self._ece_count.sum().clamp_min(1.0)
        mean_probability = torch.zeros_like(self._ece_probability_sum)
        observed_positive = torch.zeros_like(self._ece_positive_sum)
        mean_probability[nonempty] = (
            self._ece_probability_sum[nonempty] / self._ece_count[nonempty]
        )
        observed_positive[nonempty] = (
            self._ece_positive_sum[nonempty] / self._ece_count[nonempty]
        )
        ece = torch.sum(
            (self._ece_count / total_pixels) * torch.abs(mean_probability - observed_positive)
        )
        logs['diag_ece15'] = float(ece.item())

        threshold_means = self._threshold_iou_sum / count
        best_index = int(torch.argmax(threshold_means).item())
        logs['diag_best_threshold'] = self.probability_thresholds[best_index]
        logs['diag_best_threshold_iou'] = float(threshold_means[best_index].item())
        for threshold, mean_iou in zip(self.probability_thresholds, threshold_means):
            key = f'diag_iou_at_p{threshold:.2f}'.replace('.', '_')
            logs[key] = float(mean_iou.item())

        return logs
