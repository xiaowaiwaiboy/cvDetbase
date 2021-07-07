import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_Attention(nn.Module):
    def __init__(self, bn=True):
        super(Self_Attention, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(2, 2))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))

        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(32)

        self.Cv1 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))

        self.cv2 = nn.Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
        self.cv3 = nn.Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, under, over):
        x = torch.cat((under, over), dim=1)
        output = self.relu(self.bn(self.conv1(x)))
        output = self.maxpool(output)
        output = self.relu(self.bn2(self.conv2(output)))

        C = self.Cv1(output)
        C = C.view(C.shape[0] * C.shape[1], C.shape[2] * C.shape[3])

        c1 = self.cv2(output)
        c1 = c1.view(c1.shape[0] * c1.shape[2] * c1.shape[3], 8)

        c2 = self.cv3(output)
        c2 = c2.view(c2.shape[0] * c2.shape[2] * c2.shape[3], 8).t()

        c = torch.nn.Softmax(dim=1)(torch.mm(c1, c2))

        c = c.view(output.shape[0], c.shape[0], int(c.shape[1] // output.shape[0]))

        c = c.view(c.shape[0] * c.shape[1], c.shape[2])

        attention_map = torch.mm(C, c.t())

        attention_map = attention_map.view(output.shape[0], output.shape[1], output.shape[2] * output.shape[0],
                                           output.shape[3] * output.shape[0])

        attention_map = F.interpolate(attention_map, size=[under.shape[2], under.shape[3]])

        return attention_map


if __name__ == '__main__':
    model = Self_Attention()
    print(model)
    under, over = torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
    att = model(under, over)
    print(att.shape)
