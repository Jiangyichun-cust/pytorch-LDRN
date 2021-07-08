import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class basicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand):
        super(basicConv, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.expand_conv = nn.Conv2d(in_channels, expand * in_channels, 1, bias=False)
        self.conv1 = nn.Conv2d(expand * in_channels, expand * in_channels, kernel_size, 1, kernel_size // 2, groups= expand * in_channels, bias=False)
        self.impress_conv = nn.Conv2d(expand * in_channels, in_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size // 2, groups= in_channels, bias=False)
        self.output = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    def forward(self, x):
        x = self.lrelu(self.expand_conv(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.impress_conv(x))
        x = self.lrelu(self.conv2(x))
        return self.output(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = basicConv(in_channels, in_channels, 3, 2)
        self.conv2 = basicConv(in_channels, in_channels, 3, 2)

        self.sc_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sc_weight.data.fill_(0.25)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 * x + x2 + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, wide=64, block_num=16):
        super(Generator, self).__init__()
        rgb_mean = (0.4746, 0.4540, 0.4100)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)
        self.add_mean = MeanShift(1, rgb_mean, rgb_std, 1)

        self.blocknum = block_num
        self.conv1 = nn.Conv2d(in_channels, wide, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(wide, wide, 3, padding=1, bias=False)
        self.FMSRB = nn.ModuleList()
        for i in range(block_num):
            self.FMSRB.append(BasicBlock(wide, wide))

        self.infopool = nn.Conv2d(block_num * wide, wide, 1, bias=False)
        self.conv3 = nn.Conv2d(wide, wide, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(wide, out_channels * 16, 3, padding=1, bias=False)
        self.upscale = nn.PixelShuffle(4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xo = self.sub_mean(x)
        x = self.relu(self.conv1(xo))
        x = self.conv2(x)
        x1 = x
        for i in range(self.blocknum):
            x = self.FMSRB[i](x)
            if i == 0:
                x2 = x
            else:
                x2 = torch.cat([x2, x], dim=1)
        x = self.infopool(x2)
        x = x + x1
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upscale(x)
        x = self.add_mean(x)
        return x






