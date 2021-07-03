import torch.nn as nn
import torch.nn.functional as F
import math
from CNN.model.utils import Conv_Module, BACKBONES, build_attention


class ResNeXt_Block(nn.Module):
    def __init__(self, in_chnls, cardinality, group_depth, stride, is_se=False, norm=None):
        super(ResNeXt_Block, self).__init__()
        self.is_se = is_se
        self.group_chnls = cardinality * group_depth
        self.conv = nn.Sequential(
            Conv_Module(in_chnls, self.group_chnls, (1, 1), (1, 1), (0, 0), activation=nn.ReLU6(inplace=True),
                        norm=norm),
            Conv_Module(self.group_chnls, self.group_chnls, (3, 3), (stride, stride), (1, 1), groups=cardinality,
                        activation=nn.ReLU6(inplace=True), norm=norm),
            Conv_Module(self.group_chnls, self.group_chnls * 2, (1, 1), (1, 1), (0, 0), norm=norm)
        )
        if self.is_se:
            self.se = build_attention(cfg='SE', in_chnls=self.group_chnls * 2, ratio=16)
        self.short_cut = Conv_Module(in_chnls, self.group_chnls * 2, (1, 1), (stride, stride), (0, 0), norm=norm)

    def forward(self, x):
        out = self.conv(x)
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += self.short_cut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    def __init__(self,
                 layers,
                 cardinality,
                 group_depth,
                 num_classes,
                 is_se=False,
                 include_top=False,
                 out_indices=None,
                 norm=None,
                 **kwargs):
        super(ResNeXt, self).__init__()
        self.is_se = is_se
        self.if_include_top = include_top
        self.cardinality = cardinality
        self.channels = 64
        if out_indices is None:
            self.out_indices = [3, 4, 5]
        else:
            self.out_indices = out_indices
        self.features = nn.ModuleList([Conv_Module(3, self.channels, 7, 2, 3, activation=nn.ReLU6(inplace=True), norm=norm)])
        self.features.append(nn.MaxPool2d(3, 2, 1))
        self.return_features_channels = []
        for i, n in enumerate(layers):
            d = pow(2, i) * group_depth
            self.features.append(self.make_layers(d, n, stride=1 if i == 0 else 2, norm=norm))

        self.fc = nn.Linear(self.channels, num_classes)  # 224x224 input size
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layers(self, d, blocks, stride, norm):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride, self.is_se, norm))
            self.channels = self.cardinality * d * 2
        if len(self.features) in self.out_indices:
            self.return_features_channels.append(self.channels)
        return nn.Sequential(*layers)

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                res.append(x)
        if self.if_include_top:
            cls = F.avg_pool2d(x, 7)
            cls = cls.view(cls.size(0), -1)
            cls = F.softmax(self.fc(cls))
            return cls
        else:
            return res

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False


@BACKBONES.register_module()
class resNeXt50_32x4d(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(resNeXt50_32x4d, self).__init__()
        self.model = ResNeXt([3, 4, 6, 3], 32, 4, num_classes, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class resNeXt101_32x4d(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(resNeXt101_32x4d, self).__init__()
        self.model = ResNeXt([3, 4, 23, 3], 32, 4, num_classes, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class resNeXt101_64x4d(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(resNeXt101_64x4d, self).__init__()
        self.model = ResNeXt([3, 4, 23, 3], 64, 4, num_classes, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)
