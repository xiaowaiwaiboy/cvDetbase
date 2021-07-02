import torch.nn as nn
import torch.nn.functional as F
from cnn_model.model.utils import NECKS, Conv_Module


@NECKS.register_module()
class FPN(nn.Module):
    def __init__(self, out_channel=256, use_p5=True, in_channels=None, **kwargs):
        super(FPN, self).__init__()
        if in_channels is None:
            self.in_channels = [64, 128, 256, 512, 1024, 2048]
        else:
            self.in_channels = in_channels
        self.out_channel = out_channel
        self.lateral_5 = nn.Conv2d(self.in_channels[-1], out_channel, kernel_size=(1, 1))
        self.lateral_4 = nn.Conv2d(self.in_channels[-2], out_channel, kernel_size=(1, 1))
        self.lateral_3 = nn.Conv2d(self.in_channels[-3], out_channel, kernel_size=(1, 1))

        self.conv_5 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv_4 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv_3 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
        if use_p5:
            self.conv_out6 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        else:
            self.conv_out6 = nn.Conv2d(self.fpn_sizes[-1], out_channel, kernel_size=(3, 3), padding=(1, 1),
                                       stride=(2, 2))
        self.conv_out7 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)

    @staticmethod
    def upsamplelike(inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                             mode='nearest')

    @staticmethod
    def init_conv_kaiming(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        P5 = self.lateral_5(C5)
        P4 = self.lateral_4(C4)
        P3 = self.lateral_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        # P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5) if self.use_p5 else self.conv_out6(C5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]
