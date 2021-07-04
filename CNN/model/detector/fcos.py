import torch
import torch.nn as nn
from CNN.model.utils import build_backbone, build_neck, build_head, build_loss, build_generator, DETECTORS
import numpy as np


def coords_fmap2orig(feature, stride):
    """
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    """
    h, w = feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


class Encoder(nn.Module):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
        backbone_ = config.get('backbone_')
        neck_ = config.get('neck_')
        head_ = config.get('head_')
        self.features = build_backbone(backbone_, **kwargs)
        self.neck = build_neck(neck_, in_channels=self.features.out_channel, **kwargs)
        self.head = build_head(head_, in_channel=self.neck.out_channel, **kwargs)

    def forward(self, x):
        out = self.features(x)
        out = self.neck(out)
        cls, reg, cnt = self.head(out)
        return cls, reg, cnt


class Decoder(nn.Module):
    def __init__(self, score_thres=None, nms_iou_thres=None, max_detection=None, strides=None, add_centerness=None):
        super().__init__()
        self.score_threshold = score_thres
        self.nms_iou_threshold = nms_iou_thres
        self.max_detection_boxes_num = max_detection
        self.strides = strides
        self.add_centerness = add_centerness

    def forward(self, inputs):
        """
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        """
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        if self.add_centerness:
            cls_scores = torch.sqrt(cls_scores * (cnt_preds.squeeze(dim=-1)))  # [batch_size,sum(_h*_w)]
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        """
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        """
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        """
        boxes: [?,4]
        scores: [?]
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        """
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        """
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        """
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        """
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


# class Decoder(nn.Module):
#     def __init__(self, score_thres, nms_iou_thres, max_detection, **kwargs):
#         super(Decoder, self).__init__()
#         self.score_thres = score_thres
#         self.nms_iou_thres = nms_iou_thres
#         self.max_detections = max_detection
#
#     def forward(self, inputs_):
#         """
#         inputs: cls_logits, reg_preds, anchors
#         (N, sum(H*W)*A, class_num)
#         (N, sum(H*W)*A, class_num)
#         (sum(H*W)*A, 4)
#         """
#         cls_logits, reg_preds, anchors = inputs_
#         batch_size = cls_logits.shape[0]
#         cls_logits = cls_logits.sigmoid_()
#
#         boxes = self.ped2ori_box(reg_preds, anchors)  # (N, sum(H*W)*A, 4)
#
#         cls_scores, cls_ind = torch.max(cls_logits, dim=2)  # (N, sum(H*W)*A)
#         cls_ind = cls_ind + 1
#         # select topK
#         max_det = min(self.max_detections, cls_logits.shape[1])  # topK
#         topk_ind = torch.topk(cls_scores, max_det, dim=-1, largest=True, sorted=True)[1]  # (N, topK)
#         cls_topk = []
#         idx_topk = []
#         box_topk = []
#         for i in range(batch_size):
#             cls_topk.append(cls_scores[i][topk_ind[i]])  # (topK,)
#             idx_topk.append(cls_ind[i][topk_ind[i]])  # (topK,)
#             box_topk.append(boxes[i][topk_ind[i]])  # (topK,4)
#         cls_topk = torch.stack(cls_topk, dim=0)  # (N,topK)
#         idx_topk = torch.stack(idx_topk, dim=0)  # (N,topK)
#         box_topk = torch.stack(box_topk, dim=0)  # (N,topK,4)
#
#         return self._post_process(cls_topk, idx_topk, box_topk)
#
#     def _post_process(self, topk_scores, topk_inds, topk_boxes):
#         """
#         topk_scores:(N,topk)
#         """
#         batch_size = topk_scores.shape[0]
#         # mask = topk_scores >= self.score_thres #(N,topK)
#         _cls_scores = []
#         _cls_idxs = []
#         _reg_preds = []
#         for i in range(batch_size):
#             mask = topk_scores[i] >= self.score_thres
#             per_cls_scores = topk_scores[i][mask]  # (?,)
#             per_cls_idxs = topk_inds[i][mask]  # (?,)
#             per_boxes = topk_boxes[i][mask]  # (?,4)
#             nms_ind = self._batch_nms(per_cls_scores, per_cls_idxs, per_boxes)
#             _cls_scores.append(per_cls_scores[nms_ind])
#             _cls_idxs.append(per_cls_idxs[nms_ind])
#             _reg_preds.append(per_boxes[nms_ind])
#
#         return torch.stack(_cls_scores, dim=0), torch.stack(_cls_idxs, dim=0), torch.stack(_reg_preds, dim=0)
#
#     def _batch_nms(self, scores, idxs, boxes):
#         """
#         scores:(?,)
#         """
#         if boxes.numel() == 0:
#             return torch.empty((0,), dtype=torch.int64, device=boxes.device)
#         max_coords = boxes.max()
#
#         offsets = idxs.to(boxes) * (max_coords + 1)  # (?,)
#         post_boxes = boxes + offsets[:, None]  # (?,4)
#
#         keep = self.box_nms(scores, post_boxes, self.nms_iou_thres)
#         return keep
#
#     def box_nms(self, scores, boxes, iou_thres):
#         if boxes.shape[0] == 0:
#             return torch.zeros(0, device=boxes.device).long()
#         x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#         order = torch.sort(scores, descending=True)[1]  # (?,)
#         keep = []
#         while order.numel() > 0:
#             if order.numel() == 1:
#                 keep.append(order.item())
#                 break
#             else:
#                 i = order[0].item()
#                 keep.append(i)
#
#                 xmin = torch.clamp(x1[order[1:]], min=float(x1[i]))
#                 ymin = torch.clamp(y1[order[1:]], min=float(y1[i]))
#                 xmax = torch.clamp(x2[order[1:]], max=float(x2[i]))
#                 ymax = torch.clamp(y2[order[1:]], max=float(y2[i]))
#
#                 inter_area = torch.clamp(xmax - xmin, min=0.0) * torch.clamp(ymax - ymin, min=0.0)
#
#                 iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-16)
#
#                 mask_ind = (iou <= iou_thres).nonzero().squeeze()
#
#                 if mask_ind.numel() == 0:
#                     break
#                 order = order[mask_ind + 1]
#         return torch.LongTensor(keep)
#
#     def ped2ori_box(self, reg_preds, anchors):
#         """
#         reg_preds: (N, sum(H*W)*A, 4)
#         anchors:(sum(H*W)*A, 4)
#         return (N, sum(H*W)*A, 4) 4:(x1,y1,x2,y2)
#         """
#         if torch.cuda.is_available():
#             mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
#             std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
#         else:
#             mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
#             std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
#         dx, dy, dw, dh = reg_preds[..., 0], reg_preds[..., 1], reg_preds[..., 2], reg_preds[..., 3]  # (N,sum(H*W)*A)
#         dx = dx * std[0] + mean[0]
#         dy = dy * std[1] + mean[1]
#         dw = dw * std[2] + mean[2]
#         dh = dh * std[3] + mean[3]
#
#         anchor_w = (anchors[:, 2] - anchors[:, 0]).unsqueeze(0)  # (1,sum(H*W)*A)
#         anchor_h = (anchors[:, 3] - anchors[:, 1]).unsqueeze(0)  # (1,sum(H*W)*A)
#         anchor_ctr_x = anchors[:, 0].unsqueeze(0) + anchor_w * 0.5  # (1,sum(H*W)*A)
#         anchor_ctr_y = anchors[:, 1].unsqueeze(0) + anchor_h * 0.5  # (1,sum(H*W)*A)
#
#         pred_ctr_x = dx * anchor_w + anchor_ctr_x  # (N,sum(H*W)*A)
#         pred_ctr_y = dy * anchor_h + anchor_ctr_y  # (N,sum(H*W)*A)
#         pred_w = torch.exp(dw) * anchor_w  # (N,sum(H*W)*A)
#         pred_h = torch.exp(dh) * anchor_h  # (N,sum(H*W)*A)
#
#         x1 = pred_ctr_x - pred_w * 0.5  # (N,sum(H*W)*A)
#         y1 = pred_ctr_y - pred_h * 0.5  # (N,sum(H*W)*A)
#         x2 = pred_ctr_x + pred_w * 0.5  # (N,sum(H*W)*A)
#         y2 = pred_ctr_y + pred_h * 0.5  # (N,sum(H*W)*A)
#         return torch.stack([x1, y1, x2, y2], dim=2)  # (N,sum(H*W)*A,4)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes


@DETECTORS.register_module()
class FCOS(nn.Module):
    def __init__(self, encoder=None, generator=None, decoder=None, **kwargs):
        super(FCOS, self).__init__()
        self.encoder = Encoder(encoder)
        self.targt_layer = build_generator(generator)
        self.loss_func = build_loss(cfg=encoder.get('loss_'))
        self.decoder = Decoder(**decoder, strides=generator['strides'])
        self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        if self.training:
            batch_images, boxes, classes = inputs
            cls_logits, reg_preds, cnt_logits = self.encoder(batch_images)
            targets = self.targt_layer([cls_logits, reg_preds, cnt_logits, boxes, classes])
            loss = self.loss_func([cls_logits, reg_preds, cnt_logits, targets])
            return loss
        else:
            batch_images = inputs
            cls_logits, reg_preds, cnt_logits = self.encoder(batch_images)
            cls_scores, cls_idxs, boxes = self.decoder([cls_logits, cnt_logits, reg_preds])
            boxes = self.clip_boxes(batch_images, boxes)
            return cls_scores, cls_idxs, boxes
