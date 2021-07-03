from CNN.model.utils import HEADS, Conv_Module, multi_apply
import torch
import torch.nn as nn
import math
from CNN.model.backbone.efficientnet.utils import Swish, Conv2dStaticSamePadding_


class SeparableConvBlock_(nn.Module):
    def __init__(self, in_channel, out_channel, norm, activation):
        super(SeparableConvBlock_, self).__init__()
        self.conv = nn.ModuleList([Conv_Module(in_channel, in_channel, 3, 1, 0, groups=in_channel)])
        self.conv.append(Conv_Module(in_channel, out_channel, 1, 1, 0))
        if norm:
            self.conv.append(nn.BatchNorm2d(num_features=out_channel, momentum=0.01, eps=1e-3))
        # if activation:
        #     self.conv.append(nn.SiLU(inplace=True))

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        return x


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding_(in_channels, in_channels,
                                                       kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding_(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


@HEADS.register_module()
class EfficientDetHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channel,
                 stacked_convs=4,
                 feat_channels=256,
                 num_anchors=9,
                 prior=0.01,
                 norm=None,
                 **kwargs):
        super(EfficientDetHead, self).__init__()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.stacked_convs = stacked_convs
        self.in_channel = in_channel
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes * num_anchors
        self.num_anchors = num_anchors
        self.prior = prior

        for i in range(stacked_convs):
            in_chn = self.in_channel if i == 0 else self.feat_channels
            self.cls_convs.append(SeparableConvBlock(in_chn, self.feat_channels, norm=True, activation=True))
            self.reg_convs.append(SeparableConvBlock(in_chn, self.feat_channels, norm=True, activation=True))
        self.cls = SeparableConvBlock(self.feat_channels, self.cls_out_channels, norm=False, activation=False)
        self.reg = SeparableConvBlock(self.feat_channels, num_anchors * 4, norm=False, activation=False)

        self.apply(self.init_conv_RandomNormal)

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, -math.log((1 - self.prior) / self.prior))

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        batch_size, channel, H, W = x.shape
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.cls(cls_feat).permute(0, 2, 3, 1).contiguous().view(batch_size, H * W * self.num_anchors, -1)
        bbox_pred = self.reg(reg_feat).permute(0, 2, 3, 1).contiguous().view(batch_size, H * W * self.num_anchors, -1)
        return cls_score, bbox_pred

    def forward(self, feats):
        cls_scores, bbox_preds = multi_apply(self.forward_single, feats)
        return torch.cat(cls_scores, dim=1), torch.cat(bbox_preds, dim=1)
