import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from CNN.model.utils import BACKBONES, Conv_Module

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm=None):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv_Module(inplanes, planes, 3, stride, 1, activation=nn.ReLU(inplace=True), norm=norm),
            Conv_Module(planes, planes, 3, stride, 1, activation=None, norm=norm)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # ResNet-B
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            Conv_Module(inplanes, planes, 1, 1, 0, activation=nn.ReLU(inplace=True), norm=norm),
            Conv_Module(planes, planes, 3, stride, 1, activation=nn.ReLU(inplace=True), norm=norm),
            Conv_Module(planes, planes * 4, 1, 1, 0, norm=norm)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=False, out_indices=None, norm=None, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.if_include_top = include_top

        if out_indices is None:
            self.out_indices = [3, 4, 5]
        else:
            self.out_indices = out_indices

        self.features = nn.ModuleList([Conv_Module(3, 64, 7, 2, 3, activation=nn.ReLU(inplace=True), norm=norm)])
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.return_features_channels = []
        for i, n in enumerate(layers):
            out_channel = 64 * pow(2, i)
            self.features.append(self._make_layer(block, out_channel, n, stride=1 if i == 0 else 2, norm=norm))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, norm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv_Module(self.inplanes, planes * block.expansion, 1, stride, 0, norm=norm)
        layers = [block(self.inplanes, planes, stride, downsample, norm=norm)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        if len(self.features) in self.out_indices:
            self.return_features_channels.append(self.inplanes)
        return nn.Sequential(*layers)

    def forward(self, x):
        res = []
        for i, n in enumerate(self.features):
            x = n(x)
            if i in self.out_indices:
                res.append(x)
        if self.if_include_top:
            x = self.avgpool(res[-1])
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
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
class resnet18(nn.Module):
    def __init__(self, pretrained=False,  **kwargs):
        super(resnet18, self).__init__()
        self.layers = [2, 2, 2, 2]
        self.block = BasicBlock
        self.pretrained = pretrained
        self.model = self.build_model(**kwargs)
        self.out_channel = self.model.return_features_channels

    def build_model(self, **kwargs):
        model = ResNet(self.block, self.layers, **kwargs)
        if self.pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        return model

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class resnet34(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(resnet34, self).__init__()
        self.layers = [3, 4, 6, 3]
        self.block = BasicBlock
        self.pretrained = pretrained
        self.model = self.build_model(**kwargs)
        self.out_channel = self.model.return_features_channels

    def build_model(self, **kwargs):
        model = ResNet(self.block, self.layers, **kwargs)
        if self.pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        return model

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class resnet50(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(resnet50, self).__init__()
        self.layers = [3, 4, 6, 3]
        self.block = Bottleneck
        self.pretrained = pretrained
        self.model = self.build_model(**kwargs)
        self.out_channel = self.model.return_features_channels

    def build_model(self, **kwargs):
        model = ResNet(self.block, self.layers, **kwargs)
        if self.pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        return model

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class resnet101(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(resnet101, self).__init__()
        self.layers = [3, 4, 23, 3]
        self.block = Bottleneck
        self.pretrained = pretrained
        self.model = self.build_model(**kwargs)
        self.out_channel = self.model.return_features_channels

    def build_model(self, **kwargs):
        model = ResNet(self.block, self.layers, **kwargs)
        if self.pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        return model

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class resnet152(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(resnet152, self).__init__()
        self.layers = [3, 8, 36, 3]
        self.block = Bottleneck
        self.pretrained = pretrained
        self.model = self.build_model(**kwargs)
        self.out_channel = self.model.return_features_channels

    def build_model(self, **kwargs):
        model = ResNet(self.block, self.layers, **kwargs)
        if self.pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
        return model

    def forward(self, x):
        return self.model(x)


# def resnet18_(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return model
#
#
# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model
#
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
#         # model.load_state_dict(torch.load('../../resnet50.pth'),strict=False)
#     return model
#
#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model
#
#
# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model

#
# if __name__ == '__main__':
#     model = resnet18(pretrained=False, block=BasicBlock)
#     print(model)
