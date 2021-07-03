from CNN.model.utils import build_loss, LOSSES, calc_iou
import torch.nn as nn
import torch


@LOSSES.register_module()
class retina_loss(nn.Module):
    def __init__(self, cls_loss, reg_loss, **kwargs):
        super(retina_loss, self).__init__()
        self.cls_loss = build_loss(cls_loss)
        self.reg_loss = build_loss(reg_loss)

    def forward(self, inputs):
        """
        cls_logits :(n, sum(H*W)*A, class_num+1)
        reg_preds:(n, sum(H*W)*A, 4)
        anchors:(sum(H*W)*A, 4)
        boxes:(n, max_num, 4)
        classes:(n, max_num)
        """
        cls_logits, reg_preds, anchors, boxes, classes = inputs
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + anchor_widths * 0.5
        anchor_ctr_y = anchors[:, 1] + anchor_heights * 0.5

        bacth_size = cls_logits.shape[0]
        class_loss = []
        reg_loss = []
        for i in range(bacth_size):
            per_cls_logit = cls_logits[i, :, :]  # (sum(H*W)*A, class_num)
            per_reg_pred = reg_preds[i, :, :]
            per_boxes = boxes[i, :, :]
            per_classes = classes[i, :]
            mask = per_boxes[:, 0] != -1
            per_boxes = per_boxes[mask]  # (?, 4)
            per_classes = per_classes[mask]  # (?,)
            if per_classes.shape[0] == 0:
                alpha_factor = torch.ones(
                    per_cls_logit.shape).cuda() * 0.25 if torch.cuda.is_available() else torch.ones(
                    per_cls_logit.shape) * 0.25
                alpha_factor = 1. - alpha_factor
                focal_weights = per_cls_logit
                focal_weights = alpha_factor * torch.pow(focal_weights, 2.0)
                bce = -(torch.log(1.0 - per_cls_logit))
                cls_loss = focal_weights * bce
                class_loss.append(cls_loss.sum())
                reg_loss.append(torch.tensor(0).float())
                continue
            IoU = calc_iou(anchors, per_boxes)  # (sum(H*W)*A, ?)

            iou_max, max_ind = torch.max(IoU, dim=1)  # (sum(H*W)*A,)

            targets = torch.ones_like(per_cls_logit) * -1  # (sum(H*W)*A, class_num)

            targets[iou_max < 0.4, :] = 0  # bg

            pos_anchors_ind = iou_max >= 0.5  # (?,)
            num_pos = torch.clamp(pos_anchors_ind.sum().float(), min=1.0)

            assigned_classes = per_classes[max_ind]  # (sum(H*W)*A, )
            assigned_boxes = per_boxes[max_ind, :]  # (sum(H*W)*A, 4)

            targets[pos_anchors_ind, :] = 0
            targets[pos_anchors_ind, (assigned_classes[pos_anchors_ind]).long() - 1] = 1

            class_loss.append(self.cls_loss(per_cls_logit, targets).view(1) / num_pos)
            reg_loss.append(self.reg_loss(pos_anchors_ind, [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
                                          assigned_boxes, per_reg_pred))
        cls_loss = torch.stack(class_loss).mean()
        reg_loss = torch.stack(reg_loss).mean()
        total_loss = cls_loss + reg_loss
        return cls_loss, reg_loss, total_loss
