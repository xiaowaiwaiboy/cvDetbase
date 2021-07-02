import math
import torch
import torch.nn as nn
from cnn_model.model.utils import BACKBONES, Conv_Module
import torch.nn.functional as F


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', norm=None):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            Conv_Module(inp, exp, 1, 1, 0, activation=nlin_layer(inplace=True), norm=norm),
            # dw
            Conv_Module(exp, exp, kernel, stride, padding, groups=exp, activation=nlin_layer(inplace=True), norm=norm),
            SELayer(channel=exp),
            # pw-linear
            Conv_Module(exp, oup, 1, 1, 0, norm=norm)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetv3(nn.Module):
    def __init__(self,
                 out_indices=None,
                 n_class=1000,
                 dropout=0.8,
                 mode='small',
                 width_mult=1.0,
                 include_top=False,
                 norm=None,
                 **kwargs):
        super(MobileNetv3, self).__init__()
        self.if_include_top = include_top
        if out_indices is None:
            self.out_indices = [3, 9, 12]
        else:
            self.out_indices = out_indices
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'RE', 1],  # 1
                [3, 64, 24, False, 'RE', 2],  # 2
                [3, 72, 24, False, 'RE', 1],  # 3
                [5, 72, 40, True, 'RE', 2],   # 4
                [5, 120, 40, True, 'RE', 1],  # 5
                [5, 120, 40, True, 'RE', 1],  # 6
                [3, 240, 80, False, 'HS', 2],  # 7
                [3, 200, 80, False, 'HS', 1],  # 8
                [3, 184, 80, False, 'HS', 1],  # 9
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],  # 15
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'RE', 2],  # 1
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],  # 11
            ]
        else:
            raise NotImplementedError
        self.return_features_channels = []
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = nn.ModuleList(
            [Conv_Module(3, input_channel, 3, 2, 1, activation=Hswish(inplace=True), norm=norm)])

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl, norm))
            input_channel = output_channel
            if len(self.features) - 1 in self.out_indices:
                self.return_features_channels.append(output_channel)

        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
        else:
            raise NotImplementedError
        self.features.append(
            Conv_Module(input_channel, last_conv, 1, 1, 0, activation=Hswish(inplace=True), norm=norm))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(
            Conv_Module(last_conv, last_channel, 1, 1, 0, activation=Hswish(inplace=True), norm=norm))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )
        self._initialize_weights()

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                res.append(x)
        if self.if_include_top:
            x = x.mean(3).mean(2)
            cls = self.classifier(x)
            return cls
        else:
            return res

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False


@BACKBONES.register_module()
class mobilenetv3(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(mobilenetv3, self).__init__()
        self.model = MobileNetv3(**kwargs)
        self.out_channel = self.model.return_features_channels
        if pretrained:
            state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
            self.model.load_state_dict(state_dict, strict=True)
            # raise NotImplementedError

    def forward(self, x):
        return self.model(x)


# def mobilenetv3(pretrained=False, **kwargs):
#     model = MobileNetv3(**kwargs)
#     if pretrained:
#         state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
#         model.load_state_dict(state_dict, strict=True)
#         # raise NotImplementedError
#     return model
