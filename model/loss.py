import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F

logger = logging.getLogger("loss")


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

    def forward(self, pred, target, mask=None):

        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)

        if mask is not None:
            mask = mask.contiguous().view(mask.size()[0], -1)
            pred = pred * mask
            target = target * mask

        a = torch.sum(pred * target)
        b = torch.sum(pred)
        c = torch.sum(target)
        d = (2 * a) / (b + c + self.eps)

        return 1 - d


# class DiceLoss(nn.Module):
#     '''
#     Loss function from https://arxiv.org/abs/1707.03237,
#     where iou computation is introduced heatmap manner to measure the
#     diversity bwtween tow heatmaps.
#     '''

#     def __init__(self, eps=1e-6):
#         super(DiceLoss, self).__init__()
#         self.eps = eps

#     def forward(self, pred: torch.Tensor, gt, mask, weights=None):
#         '''
#         pred: one or two heatmaps of shape (N, 1, H, W),
#             the losses of tow heatmaps are added together.
#         gt: (N, 1, H, W)
#         mask: (N, H, W)
#         '''
#         assert pred.dim() == 4, pred.dim()
#         return self._compute(pred, gt, mask, weights)

#     def _compute(self, pred, gt, mask, weights):
#         if pred.dim() == 4:
#             pred = pred[:, 0, :, :]
#             gt = gt[:, 0, :, :]
#         assert pred.shape == gt.shape
#         assert pred.shape == mask.shape
#         if weights is not None:
#             assert weights.shape == mask.shape
#             mask = weights * mask

#         intersection = (pred * gt * mask).sum()
#         union = (pred * mask).sum() + (gt * mask).sum() + self.eps
#         loss = 1 - 2.0 * intersection / union
#         assert loss <= 1
#         return loss


class DBLoss(nn.Module):
    """The class for implementing DBNet loss.

    This is partially adapted from https://github.com/MhLiao/DB.

    Args:
        alpha (float): The binary loss coef.
        beta (float): The threshold loss coef.
        negative_ratio (float): The ratio of positives to negatives.
        eps (float): Epsilon in the threshold loss function.
    """

    def __init__(
        self,
        alpha: float = 1,
        beta: float = 1,
        gamma: float = 1,
        downscaled: bool = False,
        negative_ratio: float = 3,
        eps: float = 1e-6,
        ce_weight: None | Tensor = None,
        # bbce_loss: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.downscaled = downscaled
        # self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.dice_loss = DiceLoss(eps=eps)
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=0)


    def balance_bce_loss(self, pred, gt, mask):

        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        negative_count = min(negative_count, int(positive_count * self.negative_ratio))
        # logger.debug(
        #     f"[Balance BCELoss] pos_cnt: {positive_count}, neg_cnt: {negative_count}"
        # )

        loss = F.binary_cross_entropy_with_logits(pred, gt.float(), reduction="none")
        positive_loss = loss * positive
        negative_loss = loss * negative

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        # logger.debug(
        #     f"[Balance BCELoss] pos_loss: {positive_loss.sum()}, neg_loss: {negative_loss.sum()}"
        # )

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + self.eps
        )

        return balance_loss

    def l1_thr_loss(self, pred, gt, mask):
        thr_loss = torch.abs((pred - gt) * mask).sum() / (mask.sum() + self.eps)
        # logger.info(
        #     f"[L1 THR Loss] l1_loss: {thr_loss.sum()}, mask_sum: {mask.sum()}")
        return thr_loss

    def forward(
        self,
        preds: Tensor,
        gt_shr: Tensor,
        gt_shr_mask: Tensor,
        gt_thr: Tensor,
        gt_thr_mask: Tensor,
    ) -> dict[Tensor]:

        if self.downscaled:
            # preds = F.interpolate(preds,
            #                       size=gt_shr.shape[-2:],
            #                       mode="nearest")
            preds = F.interpolate(
                preds, size=gt_shr.shape[-2:], mode="bilinear", align_corners=False
            )

        pred_prob = preds[:, 0, :, :]
        pred_thr = preds[:, 1, :, :]
        pred_bi = preds[:, 2, :, :]
        pred_cat = preds[:, 3:, :, :]
        # feature_sz = preds.size()

        loss_prob = self.balance_bce_loss(pred_prob, gt_shr_mask, 1)
        # loss_bi = self.balance_bce_loss(pred_bi, gt_shr, 1)

        # loss_prob = self.balance_bce_loss(pred_prob, gt_shr, gt_shr_mask)
        loss_bi = self.balance_bce_loss(pred_bi, gt_shr_mask, gt_thr_mask)

        # loss_prob = self.balance_bce_loss(pred_prob, gt_shr, gt_thr_mask)
        # loss_bi = self.balance_bce_loss(pred_bi, gt_shr, gt_thr_mask)

        # loss_prob = self.dice_loss(pred_prob, gt_shr, gt_shr_mask)
        # loss_bi = self.dice_loss(pred_bi, gt_shr, gt_shr_mask)

        loss_thr = self.l1_thr_loss(pred_thr, gt_thr, gt_thr_mask)

        loss_cat = self.ce_loss(pred_cat, gt_shr)

        results = dict(
            loss_prob=loss_prob,
            loss_thr=self.beta * loss_thr,
            loss_bi=self.alpha * loss_bi,
            loss_cat=self.gamma * loss_cat,
        )

        return results
