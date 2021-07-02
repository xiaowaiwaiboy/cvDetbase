from cnn_model.model.utils import build_loss, LOSSES, calc_iou
import torch.nn as nn
import torch


@LOSSES.register_module()
class focal_loss(object):
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, preds, targets):
        preds = preds.sigmoid()
        preds = torch.clamp(preds, min=1e-4, max=1. - 1e-4)
        if torch.cuda.is_available():
            alpha_factor = torch.ones(targets.shape).cuda() * self.alpha
        else:
            alpha_factor = torch.ones(targets.shape) * self.alpha

        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, (1. - alpha_factor))
        focal_weights = torch.where(torch.eq(targets, 1.), 1 - preds, preds)
        focal_weights = alpha_factor * torch.pow(focal_weights, self.gamma)

        bce = - (targets * torch.log(preds) + (1. - targets) * torch.log(1. - preds))
        cls_loss = focal_weights * bce

        if torch.cuda.is_available():
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss).cuda())
        else:
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))

        return cls_loss.sum()


@LOSSES.register_module()
class smooth_l1(object):
    def __call__(self, pos_inds, anchor_infos, boxes, reg_pred):
        anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = anchor_infos  # (sum(H*W)*A,)
        if pos_inds.sum() > 0:

            pos_reg_pred = reg_pred[pos_inds, :]  # (num_pos, 4)

            gt_widths = boxes[pos_inds][:, 2] - boxes[pos_inds][:, 0]
            gt_heights = boxes[pos_inds][:, 3] - boxes[pos_inds][:, 1]
            gt_ctr_x = boxes[pos_inds][:, 0] + gt_widths * 0.5
            gt_ctr_y = boxes[pos_inds][:, 1] + gt_heights * 0.5

            pos_anchor_widths = anchor_widths[pos_inds]
            pos_anchor_heights = anchor_heights[pos_inds]
            pos_anchor_ctr_x = anchor_ctr_x[pos_inds]
            pos_anchor_ctr_y = anchor_ctr_y[pos_inds]

            gt_widths = torch.clamp(gt_widths, min=1.0)
            gt_heights = torch.clamp(gt_heights, min=1.0)

            target_dx = (gt_ctr_x - pos_anchor_ctr_x) / pos_anchor_widths
            target_dy = (gt_ctr_y - pos_anchor_ctr_y) / pos_anchor_heights
            target_dw = torch.log(gt_widths / pos_anchor_widths)
            target_dh = torch.log(gt_heights / pos_anchor_heights)

            targets = torch.stack([target_dx, target_dy, target_dw, target_dh], dim=0).t()  # (num_pos,4)
            if torch.cuda.is_available():
                targets = targets / torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).cuda()
            else:
                targets = targets / torch.FloatTensor([0.1, 0.1, 0.2, 0.2])

            reg_diff = torch.abs(targets - pos_reg_pred)  # (num_pos,4)
            reg_loss = torch.where(
                torch.le(reg_diff, 1.0 / 9.0),
                0.5 * 9.0 * torch.pow(reg_diff, 2),
                reg_diff - 0.5 / 9.0
            )
            return reg_loss.mean()
        else:
            if torch.cuda.is_available():
                reg_loss = torch.tensor(0).float().cuda()
            else:
                reg_loss = torch.tensor(0).float()

            return reg_loss
