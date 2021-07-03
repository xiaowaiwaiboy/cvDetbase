import torch
import torch.nn as nn
from CNN.model.utils import HEADS, Conv_Module, multi_apply
import math


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


@HEADS.register_module()
class FCOSHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channel,
                 stacked_convs=4,
                 feat_channel=256,
                 prior=0.01,
                 cnt_on_reg=True,
                 norm=None,
                 **kwargs):
        super(FCOSHead, self).__init__()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.prior = prior
        self.class_num = num_classes
        self.cnt_on_reg = cnt_on_reg
        self.prior = prior
        for i in range(stacked_convs):
            in_chn = in_channel if i == 0 else feat_channel
            self.cls_convs.append(
                Conv_Module(in_chn, feat_channel, 3, 1, 1, activation=nn.ReLU(inplace=True), norm=norm)
            )
            self.reg_convs.append(
                Conv_Module(in_chn, feat_channel, 3, 1, 1, activation=nn.ReLU(inplace=True), norm=norm)
            )
        self.cls = Conv_Module(feat_channel, num_classes, 3, 1, 1)
        self.reg = Conv_Module(feat_channel, 4, 3, 1, 1)
        self.cnt = Conv_Module(in_channel, 1, 3, 1, 1)
        self.apply(self.init_conv_RandomNormal)
        self.scale_exp = ScaleExp(1.0)

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, -math.log((1 - self.prior) / self.prior))

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        if not self.cnt_on_reg:
            cnt_feat = self.cnt(cls_feat)
        else:
            cnt_feat = self.cnt(reg_feat)
        cls_score = self.cls(cls_feat)
        bbox_pred = self.scale_exp(self.reg(reg_feat))
        return cls_score, bbox_pred, cnt_feat

    def forward(self, feats):
        cls_scores, bbox_preds, cnt_feats = multi_apply(self.forward_single, feats)
        return cls_scores, bbox_preds, cnt_feats



