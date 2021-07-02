import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_model.model.utils import BACKBONES, Conv_Module


class Block(nn.Module):
    """
    conv1:  Depthwise conv
    conv2:  Pointwise conv
    """

    def __init__(self, in_planes, out_planes, stride=(1, 1), norm=None):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            Conv_Module(in_planes, in_planes, 3, stride, 1, groups=in_planes, activation=nn.ReLU(), norm=norm),
            Conv_Module(in_planes, out_planes, 1, 1, 0, activation=nn.ReLU(), norm=norm)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class MobileNet(nn.Module):

    def __init__(self, out_indices=None, num_classes=10, include_top=False, config=None, norm=None):
        super(MobileNet, self).__init__()
        if config is None:
            self.cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
                        512, 512, 512, 512, 512, (1024, 2), 1024]
        else:
            self.cfg = config
        if out_indices is None:
            self.out_indices = [4, 10, 12]
        else:
            self.out_indices = out_indices
        self.if_include_top = include_top
        self.toplayer = Conv_Module(3, 32, 3, 1, 1, groups=1, activation=nn.ReLU(inplace=False), norm=norm)

        self.features = nn.ModuleList([self.toplayer])
        self.return_features_channels = []

        in_planes = 32
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]

            self.features.append(Block(in_planes=in_planes, out_planes=out_planes, stride=(stride, stride), norm=norm))
            in_planes = out_planes
            if len(self.features) - 1 in self.out_indices:
                self.return_features_channels.append(out_planes)

        self.linear = nn.Linear(self.cfg[-1] if isinstance(self.cfg[-1], int) else self.cfg[-1][0], num_classes)
        self.init_weights()

    def forward(self, x):
        res = []
        for i, l in enumerate(self.features):
            x = l(x)
            if i in self.out_indices:
                res.append(x)
        if self.if_include_top:
            cls = F.avg_pool2d(x, 2)
            cls = cls.view(cls.size(1), -1)
            cls = self.linear(cls)
            return cls
        else:
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

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False


# def mobilenetv1(pretrained=False, **kwargs):
#     model = MobileNet(**kwargs)
#     if pretrained:
#         try:
#             from torch.hub import load_state_dict_from_url
#         except ImportError:
#             from torch.utils.model_zoo import load_url as load_state_dict_from_url
#         # state_dict = load_state_dict_from_url(
#         #     'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv1_1.0-f2a8633.pth.tar?dl=1', progress=True)
#         state_dict = torch.load('mobilenetv1_1.0-f2a8633.pth.tar')
#         model.load_state_dict(state_dict)
#     return model


@BACKBONES.register_module()
class mobilenetv1(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(mobilenetv1, self).__init__()
        self.model = MobileNet(**kwargs)
        self.out_channel = self.model.return_features_channels
        if pretrained:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            # state_dict = load_state_dict_from_url(
            #     'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv1_1.0-f2a8633.pth.tar?dl=1', progress=True)
            state_dict = torch.load('mobilenetv1_1.0-f2a8633.pth.tar')
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
