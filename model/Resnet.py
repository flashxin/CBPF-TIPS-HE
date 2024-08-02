import torch
from torch import nn
import math
import model.HRnet
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU()

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        # print(X.type)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        # print(F1)

        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            # nn.Conv2d()
            nn.BatchNorm2d(F1),

            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ResModel(nn.Module):

    def __init__(self, n_class, inchannel, mode='train'):
        super(ResModel, self).__init__()
        self.mode = mode
        self.stage1 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
            IndentityBlock(256, 3, [64, 64, 256]),
            IndentityBlock(256, 3, [64, 64, 256]),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
            IndentityBlock(2048, 3, [512, 512, 2048]),
            IndentityBlock(2048, 3, [512, 512, 2048]),
        )
        self.pool = nn.AvgPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(32768, n_class)
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        pool = self.pool(out)
        out = pool.view(pool.size(0), -1)
        out = self.fc(out)
        if self.mode == 'val':
            return out, pool
        return out


class BlockAttention(nn.Module):

    def __init__(self, inchannel, mode='train'):
        super(BlockAttention, self).__init__()
        self.mode = mode
        self.stage1 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
            IndentityBlock(256, 3, [64, 64, 256]),
            IndentityBlock(256, 3, [64, 64, 256]),
        )  # ouput 56*56
        self.stage3 = nn.Sequential(
            ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
        )  # output 512,28*28
        self.outstage512 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

    def forward(self, X):  # S L输出是256层次的
        out = self.stage1(X)
        out1 = self.stage2(out)
        out2 = self.stage3(out1)
        out2 = self.outstage512(out2)
        return out2, out1


class BasicBlock(nn.Module):

    def __init__(self, channel, mode='train'):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
        )

    def forward(self, x):
        return x + self.block(x)


def softmax2dim(values):
    shape = values.shape
    values = values.view(-1, shape[2] * shape[3])
    values = nn.functional.softmax(values)
    return values.view(shape)


class CrossAttention(nn.Module):

    def __init__(self, Achannel, Bchannel, mode='train'):
        super(CrossAttention, self).__init__()
        self.mode = mode
        self.Achannel = Achannel
        self.Bchannel = Bchannel
        self.AchannelAtt = BlockAttention(inchannel=self.Achannel)
        self.BchannelAtt = BlockAttention(inchannel=self.Bchannel)
        self.Channel256toA = nn.Sequential(
            nn.Conv2d(256, self.Achannel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.Achannel),
        )
        self.Channel256toB = nn.Sequential(
            nn.Conv2d(256, self.Bchannel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.Bchannel),
        )
        self.A2B28 = BasicBlock(256)
        self.B2A28 = BasicBlock(256)
        self.Softmax = nn.Softmax()
        self.up = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, Ainput, Binput):  # S L输出是256层次的
        Aout28, Aout56 = self.AchannelAtt(Ainput)
        Bout28, Bout56 = self.BchannelAtt(Binput)
        Ain56 = Aout28 + self.B2A28(Bout28)
        Bin56 = Bout28 + self.A2B28(Aout28)
        Aout56 = self.Channel256toA(Ain56)
        Bout56 = self.Channel256toB(Bin56)
        hotA = softmax2dim(Aout56)
        hotB = softmax2dim(Bout56)
        hotA=self.up(hotA)
        hotB = self.up(hotB)
        # Bout56= softmax2dim(Bout56)
        Aout56 = hotA * Ainput + Ainput
        Bout56 = hotB * Binput + Binput
        return Aout56, Bout56,hotA,hotB


class doubleAttResModel(nn.Module):
    def __init__(self, Achannel, Bchannel, kinds, mode='train'):
        super(doubleAttResModel, self).__init__()
        self.mode = mode
        self.kinds = kinds
        self.Achannel = Achannel
        self.Bchannel = Bchannel
        self.crossnet = CrossAttention(Achannel=self.Achannel, Bchannel=self.Bchannel)
        self.net = model.HRnet.HRnet(Ainchannel=self.Achannel, Binchannel=self.Bchannel, kinds=self.kinds)
        self.addchannel = nn.Sequential(
            nn.Conv2d(self.Achannel + self.Bchannel, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, Ain, Bin):
        # print(Ain.type)
        Aout, Bout,hota,hotb= self.crossnet(Ain, Bin)
        # Aout, Bout=Ain, Bin
        out,s4_0A,s4_0B = self.net(Aout, Bout)
        return out,hota,hotb


class doubleinAttResModel(nn.Module):
    def __init__(self, Achannel, Bchannel, kinds, mode='train'):
        super(doubleinAttResModel, self).__init__()
        self.mode = mode
        self.kinds = kinds
        self.Achannel = Achannel
        self.Bchannel = Bchannel
        self.Anet = ResModel(inchannel=self.Achannel, n_class=2)
        self.Bnet = ResModel(inchannel=self.Bchannel, n_class=2)
        self.Cnet = torchvision.models.densenet121()

        self.L = nn.Linear(4, 16)
        self.L1 = nn.Linear(16, self.kinds)
        self.L2 = nn.Linear(1000, self.kinds)

        self.channel = nn.Sequential(
            nn.Conv2d(20, 3, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
        )
        self.Softmax=nn.Softmax()

    def forward(self, Ain, Bin):
        Aout = self.Anet(Ain)
        Bout = self.Bnet(Bin)
        out = torch.cat([Aout, Bout], dim=1)
        out = self.L(out)
        out = self.L1(out)
        # out=nn.functional.softmax(out,dim=1)
        # out=self.channel(Ain)
        # out=self.Cnet(out)
        # out=self.L2(out)
        out = self.Softmax(out)
        # print(out)
        return out


def weight_init(m):  # 初始化权重
    if isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


if __name__ == "__main__":
    a = torch.rand([1, 1, 224, 224])
    b = torch.rand([1, 2, 224, 224])
    net = doubleAttResModel(Achannel=1, Bchannel=2)
    a = net(a, b)
    print(a.shape)
    print(a)
    # print(b.shape)
