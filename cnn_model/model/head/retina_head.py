import torch
import torch.nn as nn
from cnn_model.model.utils import HEADS, Conv_Module, multi_apply
import math


@HEADS.register_module()
class RetinaHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 feat_channels=256,
                 num_anchors=9,
                 prior=0.01,
                 norm=None,
                 **kwargs):
        super(RetinaHead, self).__init__()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.stacked_convs = stacked_convs
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes * num_anchors
        self.num_anchors = num_anchors
        for i in range(self.stacked_convs):
            inc_chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                Conv_Module(inc_chn, self.feat_channels, 3, 1, 1, activation=nn.ReLU(inplace=True), norm=norm)
            )
            self.reg_convs.append(
                Conv_Module(inc_chn, self.feat_channels, 3, 1, 1, activation=nn.ReLU(inplace=True), norm=norm)
            )
        self.retina_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, (3, 3), padding=(1, 1))
        self.retina_reg = nn.Conv2d(self.feat_channels, num_anchors * 4, (3, 3), padding=(1, 1))
        self.apply(self.init_conv_RandomNormal)
        nn.init.constant_(self.retina_cls.bias, -math.log((1 - prior) / prior))

    @staticmethod
    def init_conv_RandomNormal(module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        batch_size, channel, H, W = x.shape
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat).permute(0, 2, 3, 1).contiguous().view(batch_size, H*W*self.num_anchors, -1)
        bbox_pred = self.retina_reg(reg_feat).permute(0, 2, 3, 1).contiguous().view(batch_size, H*W*self.num_anchors, -1)
        return cls_score, bbox_pred

    def forward(self, feats):
        cls_scores, bbox_preds = multi_apply(self.forward_single, feats)
        return torch.cat(cls_scores, dim=1), torch.cat(bbox_preds, dim=1)
