#Input size 20*224*224
import torch
from torch import nn
import math
class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
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
class GlobalEncoder(nn.Module):#静脉/动脉共享权重
    def __init__(self,inchannel,outchannel,mode):
        super().__init__()
        self.mode = mode
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.stage1 = nn.Sequential(
            nn.Conv2d(self.inchannel, 64, 7, stride=2, padding=3, bias=False),
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
        self.reshape=nn.Sequential(
            nn.Conv2d(1024, self.outchannel, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.outchannel),
            nn.ReLU(True),
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out=  self.reshape(out)
        return out
class Endlayer(nn.Module):#输入大小为512*14*14
    def __init__(self,inchannel,outchannel,mode):
        super().__init__()
        self.mode = mode
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.layers=nn.Sequential(
            IndentityBlock(self.inchannel, 3, [self.inchannel, 1024, self.inchannel]),
            nn.Conv2d(self.inchannel, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            IndentityBlock(256, 3, [256, 512, 256]),
            nn.Conv2d(256, self.outchannel, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.outchannel),
            nn.ReLU(True),
            IndentityBlock(self.outchannel, 3, [self.outchannel, 64, self.outchannel]),
        )#128*14*14

    def forward(self,x):
        return self.layers(x)

class RPHE(nn.Module):
    def __init__(self,inchannel,outchannel,kind,mode):
        super(RPHE,self).__init__()
        self.inchannel=inchannel
        self.kind=kind
        self.mode=mode
        self.outchannel = outchannel
        self.GlobalEncoder=GlobalEncoder(inchannel=self.inchannel,outchannel=self.outchannel,mode=self.mode)
        self.HEEncoder=Endlayer(inchannel=512,outchannel=128,mode=self.mode)
        self.edemaEncoder = Endlayer(inchannel=512, outchannel=128, mode=self.mode)
        self.HEFC=nn.Sequential(
            nn.Linear(128*14*14,256),
            nn.Linear(256,self.kind),
        )
        self.edemaFC=nn.Sequential(
            nn.Linear(128*14*14,256),
            nn.Linear(256,self.kind),
        )
        self.GlobalEncoder1 = GlobalEncoder(inchannel=self.inchannel, outchannel=self.outchannel, mode=self.mode)
    def forward(self,x_base,x_add):
        y_base=self.GlobalEncoder(x_base)
        y_add=self.GlobalEncoder1(x_add)
        out_global=y_add-y_base
        out_base=y_base+y_add
        out=torch.cat([y_base,y_add],dim=1)#256*2*14*14 拼接

        HEout=self.HEEncoder(out)

        edema=self.edemaEncoder(out)

        HEout = HEout.view(HEout.size(0), -1)
        HEout = self.HEFC(HEout)
        HEout=nn.functional.softmax(HEout)

        edema = edema.view(HEout.size(0), -1)
        edema = self.edemaFC(edema)
        edema = nn.functional.softmax(edema)
        return HEout,edema


if __name__ == "__main__":
    x=torch.rand([1,1,224,224])
    net=RPHE(inchannel=1,outchannel=256,kind=2,mode='train')
    out1,out2=net(x,x)
    print(out1.shape)   #outchannel*14*14
    print(out2.shape)   #outchannel*14*14