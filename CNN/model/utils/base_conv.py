import torch.nn as nn


class Conv_Module(nn.Module):
    """
    Conv2d Module. options: activation、normlayer
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: tuple or int,
                 stride: tuple or int,
                 padding: tuple or int,
                 dilation=(1, 1),
                 groups=1,
                 bias=False,
                 activation=None,
                 norm=None):
        super(Conv_Module, self).__init__()
        layer = [nn.Conv2d(in_channel, out_channel,
                           kernel_size=kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size),
                           stride=stride if isinstance(stride, tuple) else (stride, stride),
                           padding=padding if isinstance(padding, tuple) else (padding, padding),
                           dilation=dilation, groups=groups, bias=bias
                           )]
        if norm is not None and norm == 'BN':
            layer.append(nn.BatchNorm2d(out_channel))
        elif norm is not None and norm == 'GN':
            layer.append(nn.GroupNorm(num_groups=32, num_channels=in_channel))
        elif norm is not None and norm not in ['BN', 'GN']:
            raise KeyError
        if activation is not None:
            layer.append(activation)
        self.seq = nn.Sequential(*layer)

    def forward(self, x):
        return self.seq(x)
