import math
import torch.nn as nn
from cnn_model.model.utils import BACKBONES, Conv_Module


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                Conv_Module(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, activation=nn.ReLU6(inplace=True), norm=norm),
                Conv_Module(hidden_dim, oup, 1, 1, 0, groups=1, norm=norm)
            )
        else:
            self.conv = nn.Sequential(
                Conv_Module(inp, hidden_dim, 1, 1, 0, activation=nn.ReLU6(inplace=True), norm=norm),
                Conv_Module(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, activation=nn.ReLU6(inplace=True), norm=norm),
                Conv_Module(hidden_dim, oup, 1, 1, 0, norm=norm)
            )

    def forward(self, x):
        if self.use_res_connect:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        return x


class MobileNetv2(nn.Module):
    def __init__(self, out_indices=None, width_multiplier=1., config=None, include_top=False, norm=None, **kwargs):
        super(MobileNetv2, self).__init__()
        block = InvertedResidual
        in_planes = int(32 * width_multiplier)
        if config is None:
            config = [
                # t, c, n, s
                [1, 16, 1, 1],  # 1
                [6, 24, 2, 2],  # 3
                [6, 32, 3, 2],  # 6
                [6, 64, 5, 2],  # 11
                [6, 96, 3, 1],  # 14
                [6, 160, 4, 2],  # 18
                [6, 320, 1, 1],  # 19
                # [3, 1280, 1, 1]
            ]
        if out_indices is None:
            self.out_indices = [6, 11, 18]
        else:
            self.out_indices = out_indices
        self.if_include_top = include_top
        self.return_features_channels = []

        self.features = nn.ModuleList([Conv_Module(3, in_planes, 3, 2, 1, activation=nn.ReLU6(inplace=True), norm=norm)])

        for t, c, n, s in config:
            output_channel = make_divisible(c * width_multiplier) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(in_planes, output_channel, s, expand_ratio=t, norm=norm))
                else:
                    self.features.append(block(in_planes, output_channel, 1, expand_ratio=t, norm=norm))
                in_planes = output_channel
                if len(self.features) - 1 in self.out_indices:
                    self.return_features_channels.append(output_channel)
        self.init_weights()

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                res.append(x)
        return res

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# def mobilenetv2(pretrained=True, **kwargs):
#     model = MobileNetv2(width_multiplier=1., **kwargs)
#     if pretrained:
#         try:
#             from torch.hub import load_state_dict_from_url
#         except ImportError:
#             from torch.utils.model_zoo import load_url as load_state_dict_from_url
#         state_dict = load_state_dict_from_url(
#             'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
#         # state_dict = torch.load('mobilenetv2_1.0-f2a8633.pth.tar')
#         model.load_state_dict(state_dict)
#     return model


@BACKBONES.register_module()
class mobilenetv2(nn.Module):
    def __init__(self, pretrained=False, width_multiplier=1., **kwargs):
        super(mobilenetv2, self).__init__()
        self.model = MobileNetv2(width_multiplier=width_multiplier, **kwargs)
        self.out_channel = self.model.return_features_channels
        if pretrained:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            state_dict = load_state_dict_from_url(
                'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
            # state_dict = torch.load('mobilenetv2_1.0-f2a8633.pth.tar')
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
