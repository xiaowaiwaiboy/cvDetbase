import torch.nn as nn
import torch.nn.functional as F
import math
from CNN.model.utils import BACKBONES, Conv_Module, build_attention


class Dark_block(nn.Module):
    """block for darknet"""
    def __init__(self, channels, is_se=False, inner_channels=None, norm=None):
        super(Dark_block, self).__init__()
        self.is_se = is_se
        if inner_channels is None:
            inner_channels = channels // 2
        self.conv = nn.Sequential(
            Conv_Module(channels, inner_channels, 1, 1, 0, activation=nn.LeakyReLU(inplace=True), norm=norm),
            Conv_Module(inner_channels, channels, 3, 1, 1, norm=norm)
        )
        if self.is_se:
            self.se = build_attention(cfg='SE', in_chnls=channels, ratio=16)

    def forward(self, x):
        out = self.conv(x)
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += x
        return F.leaky_relu(out)


class DarkNet(nn.Module):

    def __init__(self, layers, num_classes=1000, is_se=False, include_top=False, out_indices=None, norm=None, **kwargs):
        super(DarkNet, self).__init__()
        self.is_se = is_se
        self.if_include_top = include_top
        if out_indices is None:
            self.out_indices = [2, 4, 6, 8, 10]
        else:
            self.out_indices = out_indices
        filters = [64, 128, 256, 512, 1024]
        self.return_features_channels = []
        self.features = nn.ModuleList([Conv_Module(3, 32, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm)])
        self.features.append(Conv_Module(32, 64, 3, 2, 1, activation=nn.ReLU6(inplace=True), norm=norm))
        for i in range(len(layers) - 1):
            self.features.append(self.make_layers(filters[i], layers[i], norm)),
            self.features.append(Conv_Module(filters[i], filters[i + 1], 3, 2, 1, activation=nn.ReLU6(inplace=True), norm=norm))
        self.features.append(self.make_layers(filters[-1], layers[-1], norm))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[4], num_classes)
        self.init_weights()

    def make_layers(self, num_filter, num_layers, norm):
        layers = []
        for _ in range(num_layers):
            layers.append(Dark_block(num_filter, self.is_se, norm=norm))
        if len(self.features) in self.out_indices:
            self.return_features_channels.append(num_filter)
        return nn.Sequential(*layers)

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                res.append(x)
        if self.if_include_top:
            cls = self.global_pool(x)
            cls = cls.view(cls.size(0), -1)
            cls = self.fc(cls)
            cls = F.softmax(cls)
            return cls
        else:
            return res

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False


@BACKBONES.register_module()
class darknet_53(nn.Module):
    def __init__(self, **kwargs):
        super(darknet_53, self).__init__()
        self.model = DarkNet([1, 2, 8, 8, 4], **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)

# def darknet_53(**kwargs):
#     return DarkNet([1, 2, 8, 8, 4], **kwargs)
