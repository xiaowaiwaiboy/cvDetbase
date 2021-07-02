from cnn_model.model.utils import GENERATORS
import numpy as np
import torch.nn as nn
import torch


def coords_fmap2orig(image_shape, stride):
    """
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    """
    h, w = image_shape
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y, shift_x, shift_y], -1) + stride // 2
    return coords


@GENERATORS.register_module()
class Horizontal_Generator(nn.Module):
    def __init__(self, ratios, scales, strides, sizes):
        super().__init__()
        # if config is None:
        #     self.config = {'strides': [8, 16, 32, 64, 128], 'pyramid_levels': [3, 4, 5, 6, 7],
        #                    'sizes': [32, 64, 128, 256, 512], 'ratios': [0.5, 1, 2],
        #                    'scales': [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]}
        # else:
        #     self.config = config

        self.ratios = np.array(ratios)
        self.scales = np.array(eval(scales))
        self.size = sizes
        self.strides = strides

    def forward(self, image):
        H, W = image.size(2), image.size(3)  # (ori_H, ori_W)
        feature_size = [(H / stride, W / stride) for stride in self.strides]
        all_anchors = []
        for i in range(len(feature_size)):
            anchors = self.generate_anchors(self.size[i], self.ratios, self.scales)
            shift_anchors = self.shift(anchors, feature_size[i], self.strides[i])  # (H*W, A, 4)
            all_anchors.append(shift_anchors)
        all_anchors = torch.cat(all_anchors, dim=0)
        return all_anchors

    def generate_anchors(self, base_size=16, ratios=None, scales=None):
        if ratios is None:
            ratios = np.array([0.5, 1, 2])
        if scales is None:
            scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        num_anchors = len(ratios) * len(scales)  # 9
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]  # (9,)
        # fix the ratios of w, h
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))  # (9,)
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))  # (9,)

        # transfrom from(0 ,0, w, h ) to ( x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        anchors = torch.from_numpy(anchors).float().cuda() if torch.cuda.is_available() else torch.from_numpy(
            anchors).float()
        return anchors

    def shift(self, anchors, image_shape, stride):
        """
        anchors : Tensor(num, 4)
        image_shape : (H, W)
        return shift_anchor: (H*W*num,4)
        """

        ori_coords = coords_fmap2orig(image_shape, stride)  # (H*W, 4) 4:(x,y,x,y)
        ori_coords = ori_coords.to(device=anchors.device)
        shift_anchor = ori_coords[:, None, :] + anchors[None, :, :]
        return shift_anchor.reshape(-1, 4)
