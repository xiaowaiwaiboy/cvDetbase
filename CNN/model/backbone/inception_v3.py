import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN.model.utils import BACKBONES, Conv_Module
import math


class Block_bank(nn.Module):
    """
    inception structures
    """

    def __init__(self, cfg_, norm):
        super(Block_bank, self).__init__()
        block_type, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4 = cfg_
        self.block_type = block_type  # controlled by strings "type1", "type2", "type3", "type4", "type5"
        self.block = nn.ModuleList([])
        if self.block_type == 'type1':
            self.block.append(
                nn.Sequential(*[Conv_Module(in_channels, b1_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                Conv_Module(b1_reduce, b1, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm),
                                Conv_Module(b1, b1, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm)]))
            self.block.append(
                nn.Sequential(*[Conv_Module(in_channels, b2_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                Conv_Module(b2_reduce, b2, 5, 1, 2, activation=nn.ReLU6(inplace=True), norm=norm)]))
            self.block.append(
                nn.Sequential(*[nn.AvgPool2d(3, 1, 1),
                                Conv_Module(in_channels, b3, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm)]
                              ))
            self.block.append(Conv_Module(in_channels, b4, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm))
        elif self.block_type == 'type2':
            self.block.append(
                nn.Sequential(*[Conv_Module(in_channels, b1_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                              Conv_Module(b1_reduce, b1, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm),
                              Conv_Module(b1, b1, 3, 2, 0, activation=nn.ReLU6(inplace=True), norm=norm)]))
            self.block.append(Conv_Module(in_channels, b2, 3, 2, 0, activation=nn.ReLU6(inplace=True), norm=norm))
            self.block.append(nn.MaxPool2d(3, 2, 0))
        elif self.block_type == 'type3':
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b1_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1_reduce, b1_reduce, (7, 1), (1, 1), (3, 0), activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1_reduce, b1_reduce, (1, 7), (1, 1), (0, 3),
                                            activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1_reduce, b1_reduce, (7, 1), (1, 1), (3, 0),
                                            activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1_reduce, b1, (1, 7), (1, 1), (0, 3), activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b2_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b2_reduce, b2_reduce, (1, 7), (1, 1), (0, 3),
                                            activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b2_reduce, b2, (7, 1), (1, 1), (3, 0), activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.Sequential(*[
                                  nn.AvgPool2d(3, 1, 1),
                                  Conv_Module(in_channels, b3, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(Conv_Module(in_channels, b4, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm))
        elif self.block_type == 'type4':
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b1_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1_reduce, b1, (1, 7), (1, 1), (0, 3), activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1, b1, (7, 1), (1, 1), (3, 0), activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1, b1, 3, 2, 0, activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b2_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b2_reduce, b2, 3, 2, 0, activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.MaxPool2d(3, 2, 0))
        elif self.block_type == 'type5':
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b1_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1_reduce, b1, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1, b1, (1, 3), (1, 1), (0, 1), activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b1_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1_reduce, b1, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b1, b1, (3, 1), (1, 1), (1, 0), activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b2_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b2_reduce, b2, (1, 3), (1, 1), (0, 1), activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.Sequential(*[
                                  Conv_Module(in_channels, b2_reduce, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                                  Conv_Module(b2_reduce, b2, (3, 1), (1, 1), (1, 0), activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(nn.Sequential(*[
                                  nn.AvgPool2d(3, 1, 1),
                                  Conv_Module(in_channels, b3, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm)
                              ]))
            self.block.append(Conv_Module(in_channels, b4, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm))
        else:
            raise KeyError

    def forward(self, x):
        out = []
        for i, m in enumerate(self.block):
            out_ = m(x)
            out.append(out_)
        return torch.cat(tuple(out), 1)


@BACKBONES.register_module()
class Inception_v3(nn.Module):
    def __init__(self, num_classes=1000, include_top=False, out_indices=None, norm=None, **kwargs):
        super(Inception_v3, self).__init__()
        if out_indices is None:
            self.out_indices = [3, 4, 9, 11]
        else:
            self.out_indices = out_indices
        self.if_include_top = include_top
        cfg = [
            ["type1", 192, 64, 96, 48, 64, 32, 64],  # 1
            ["type1", 256, 64, 96, 48, 64, 64, 64],  # 2
            ["type1", 288, 64, 96, 48, 64, 64, 64],
            ["type2", 288, 64, 96, 288, 384, 288, 288],
            ["type3", 768, 128, 192, 128, 192, 192, 192],
            ["type3", 768, 160, 192, 160, 192, 192, 192],
            ["type3", 768, 160, 192, 160, 192, 192, 192],
            ["type3", 768, 192, 192, 192, 192, 192, 192],
            ["type4", 768, 192, 192, 192, 320, 288, 288],
            ["type5", 1280, 448, 384, 384, 384, 192, 320],
            ["type5", 2048, 448, 384, 384, 384, 192, 320]  # 11
        ]
        first_layer = [Conv_Module(3, 32, 3, 2, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                       Conv_Module(32, 32, 3, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                       Conv_Module(32, 64, 3, 1, 1, activation=nn.ReLU6(inplace=True), norm=norm),
                       nn.MaxPool2d(3, 2, 0),
                       Conv_Module(64, 80, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                       Conv_Module(80, 192, 3, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                       nn.MaxPool2d(3, 2, 0)]
        self.features = nn.ModuleList([])
        self.features.append(nn.Sequential(*first_layer))
        self.out_channels = [c[1] for c in cfg]
        self.out_channels.append(cfg[-1][1])
        self.out_channel = [self.out_channels[i] for i in self.out_indices]
        for i in range(len(cfg)):
            self.features.append(Block_bank(cfg[i], norm))
        self.fc = nn.Linear(2048, num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                res.append(x)
        if self.if_include_top:
            cls = F.avg_pool2d(x, 8)
            cls = F.dropout(cls, 0.2, training=self.training)
            cls = cls.view(cls.size()[0], -1)
            cls = self.fc(cls)
            cls = F.softmax(cls)
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
