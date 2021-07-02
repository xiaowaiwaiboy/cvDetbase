import torch.nn as nn
import torch.nn.functional as F
from cnn_model.model.utils import Conv_Module, BACKBONES
import torch
import math


class DenseBlock(nn.Module):

    def __init__(self, input_channels, num_layers, growth_rate, norm):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self._make_layers(norm)

    def _make_layers(self, norm):
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(
                Conv_Module(self.k0 + i * self.k, 4 * self.k, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                Conv_Module(4 * self.k, self.k, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm)
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        return out


class CSP_DenseBlock(nn.Module):

    def __init__(self, in_channels, num_layers, k, part_ratio=0.5, norm=None):
        super(CSP_DenseBlock, self).__init__()
        self.part1_chnls = int(in_channels * part_ratio)
        self.part2_chnls = in_channels - self.part1_chnls
        self.dense = DenseBlock(self.part2_chnls, num_layers, k, norm)
        # trans_chnls = self.part2_chnls + k * num_layers
        # self.transtion = BN_Conv2d(trans_chnls, trans_chnls, 1, 1, 0)

    def forward(self, x):
        part1 = x[:, :self.part1_chnls, :, :]
        part2 = x[:, self.part1_chnls:, :, :]
        part2 = self.dense(part2)
        # part2 = self.transtion(part2)
        out = torch.cat((part1, part2), 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, layers, k, theta, num_classes=1000, part_ratio=None, include_top=False, out_indices=None, norm=None):
        super(DenseNet, self).__init__()
        if out_indices is None:
            self.out_indices = [2, 4, 6, 8]
        else:
            self.out_indices = out_indices
        self.if_include_top = include_top
        self.layers = layers
        self.theta = theta
        self.patches = 0
        self.block = DenseBlock if part_ratio is None else CSP_DenseBlock
        self.features = nn.ModuleList([Conv_Module(3, 2*k, 7, 2, 3, activation=nn.ReLU6(inplace=True), norm=norm)])
        self.features.append(nn.MaxPool2d(3, 2, 1))
        self.return_features_channels = []
        self.out_channel = 2 * k
        for i in range(len(self.layers)):
            self.features.append(self.block(self.out_channel, self.layers[i], k))
            self.patches = self.out_channel + self.layers[i] * k
            if i != len(self.layers) - 1:
                transition, self.out_channel = self.make_transition(self.patches, norm)
                self.features.append(transition)
            if len(self.features) - 2 or len(self.features) - 1 in self.out_indices:
                self.return_features_channels.append(self.patches)

        self.fc = nn.Linear(self.patches, num_classes)
        self.init_weights()

    def make_transition(self, in_chls, norm):
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(
            Conv_Module(in_chls, out_chls, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
            nn.AvgPool2d(2)
        ), out_chls

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
class densenet_121(nn.Module):
    def __init__(self, k=32, theta=0.5, **kwargs):
        super(densenet_121, self).__init__()
        self.model = DenseNet([6, 12, 24, 16], k, theta, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class densenet_169(nn.Module):
    def __init__(self, k=32, theta=0.5, **kwargs):
        super(densenet_169, self).__init__()
        self.model = DenseNet([6, 12, 32, 32], k, theta, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class densenet_201(nn.Module):
    def __init__(self, k=32, theta=0.5, **kwargs):
        super(densenet_201, self).__init__()
        self.model = DenseNet([6, 12, 48, 32], k, theta, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class densenet_264(nn.Module):
    def __init__(self, k=32, theta=0.5, **kwargs):
        super(densenet_264, self).__init__()
        self.model = DenseNet([6, 12, 64, 48], k, theta, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class csp_densenet_121(nn.Module):
    def __init__(self, k=32, theta=0.5, part_ratio=0.5, **kwargs):
        super(csp_densenet_121, self).__init__()
        self.model = DenseNet([6, 12, 24, 16], k, theta, part_ratio=part_ratio, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class csp_densenet_169(nn.Module):
    def __init__(self, k=32, theta=0.5, part_ratio=0.5, **kwargs):
        super(csp_densenet_169, self).__init__()
        self.model = DenseNet([6, 12, 32, 32], k, theta, part_ratio=part_ratio, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class csp_densenet_201(nn.Module):
    def __init__(self, k=32, theta=0.5, part_ratio=0.5, **kwargs):
        super(csp_densenet_201, self).__init__()
        self.model = DenseNet([6, 12, 48, 32], k, theta, part_ratio=part_ratio, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class csp_densenet_264(nn.Module):
    def __init__(self, k=32, theta=0.5, part_ratio=0.5, **kwargs):
        super(csp_densenet_264, self).__init__()
        self.model = DenseNet([6, 12, 64, 48], k, theta, part_ratio=part_ratio, **kwargs)
        self.out_channel = self.model.return_features_channels

    def forward(self, x):
        return self.model(x)
