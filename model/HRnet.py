import wandb
import torch.utils.data as Data #批处理模块
import torch
from torch.autograd.gradcheck import gradcheck
import torchvision.datasets as dset
import copy
from torchvision import transforms
from torch import nn
import math
import numpy as np
from sklearn import linear_model
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn.functional as F
import  os
# import net.UNet as UNet
# import net.VIT as VIT
class basicBlock(nn.Module):
    def __init__(self,C):
        super(basicBlock, self).__init__()
        self.conv3x3=nn.Sequential(
            nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            # nn.ReLU(True),
        )
        self.R=nn.ReLU(True)
    def forward(self,X):
        X_shortcut = X
        X = self.conv3x3(X)
        out=X+X_shortcut
        out=self.R(out)
        return out

class Stage1Block(nn.Module):
    def __init__(self, C):
        super(Stage1Block, self).__init__()
        self.convUnit = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=1),
            nn.BatchNorm2d(C),
            nn.ReLU(True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(True),
            nn.Conv2d(C, C, kernel_size=1, stride=1),
            nn.BatchNorm2d(C),
            nn.ReLU(True),
        )

    def forward(self, X):
        X_shortcut = X
        X = self.convUnit(X)
        out = X + X_shortcut
        return out
class Stage2Tran(nn.Module):
    def __init__(self):
        super(Stage2Tran, self).__init__()
        self.convUnit1 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),#64*64*32
        )
        self.convUnit2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),#32*32*64
        )

    def forward(self, X):
        X1=self.convUnit1(X)
        X2=self.convUnit2(X)
        return X1,X2

class Stage2Fusion(nn.Module):
    def __init__(self):
        super(Stage2Fusion, self).__init__()
        self.R=nn.ReLU(True)
        self.convUnit1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#32*32*64
        )
        self.convUnit2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )

    def forward(self, X1,X2):
        out1=self.R(self.convUnit2(X2)+X1)
        out2=self.R(self.convUnit1(X1)+X2)
        return out1,out2
class Stage3Tran(nn.Module):
    def __init__(self):
        super(Stage3Tran, self).__init__()
        self.convUnit= nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
    def forward(self, X1,X2):
        X3=self.convUnit(X2)
        return X1,X2,X3

class Stage3Fusion(nn.Module):
    def __init__(self):
        super(Stage3Fusion, self).__init__()
        self.R=nn.ReLU(True)
        self.downUnit1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.upUnit1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        self.upUnit2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=4,mode='nearest')
        )
        self.upUnit3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )

    def forward(self, X1,X2,X3):
        down1=self.downUnit1(X1)
        down2=self.downUnit2(down1)
        out1=self.R(self.upUnit2(X3)+self.upUnit1(X2)+X1)
        out2=self.R(self.upUnit3(X3)+down1+X2)
        out3=self.R(down2+self.downUnit3(X2)+X3)
        return out1,out2,out3

class Stage4Tran(nn.Module):
    def __init__(self):
        super(Stage4Tran, self).__init__()
        self.convUnit= nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),#64*64*32
        )
    def forward(self, X1,X2,X3):
        X4=self.convUnit(X3)
        return X1,X2,X3,X4

class Stage4Fusion(nn.Module):
    def __init__(self):
        super(Stage4Fusion, self).__init__()
        self.R=nn.ReLU(True)
        self.downUnit1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#32*32*64
        )
        self.upUnit1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        self.upUnit2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=4,mode='nearest')
        )
        self.upUnit3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=8,mode='nearest')
        )
        self.upUnit4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        self.upUnit5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=4,mode='nearest')
        )
        self.upUnit6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )

    def forward(self, X1,X2,X3,X4):
        down1=self.downUnit1(X1)
        down2=self.downUnit2(down1)
        down3=self.downUnit3(down2)

        down4=self.downUnit4(X2)
        down5=self.downUnit5(down4)

        out1=self.R(self.upUnit3(X4)+self.upUnit2(X3)+self.upUnit1(X2)+X1)
        out2=self.R(self.upUnit5(X4)+self.upUnit4(X3)+down1+X2)
        out3=self.R(self.upUnit6(X4)+down4+down2+self.downUnit4(X2)+X3)
        out4=self.R(down5+down3+self.downUnit6(X3)+X4)
        return out1,out2,out3,out4
class  downAndEx(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(downAndEx, self).__init__()
        self.Unit=nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1),  # 64*64
            nn.BatchNorm2d(outchannel),
        )
    def forward(self,X):
        X=self.Unit(X)
        return X
class classifier1(nn.Module):#classic  classifier
    def __init__(self,kinds):
        super(classifier1, self).__init__()
        # 64*64*32 32*32*64 16*16*128 8*8*256
        self.R=nn.ReLU(True)
        self.downUnit1=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.downUnit2=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.downUnit3=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )
        self.simple=nn.Sequential(#4*4*128
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),

        )
        self.L=nn.Linear(4*4*128,kinds)
    def forward(self,X1,X2,X3,X4):
        down1=self.R(X2+self.downUnit1(X1))
        down2=self.R(X3+self.downUnit2(down1))
        down3=self.R(X4+self.downUnit3(down2))#8*8*256
        # print(down3.shape)
        # print(self.simple(down3).shape)
        out=self.simple(down3)
        # print(out.shape)
        out=F.softmax(self.L(out.view(out.size(0),-1)))
        return out
class classifier2(nn.Module):#DCt and space information
    def __init__(self,kinds):
        super(classifier2, self).__init__()
        self.Unit1=nn.Sequential(#16*16*128
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.Linear(4*4*128,kinds)
        )#8*8*256
        self.Unit2=nn.Sequential(#8*8*256
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

        )#8*8*256
        self.L=nn.Linear(4*4*64,kinds)
    def forward(self,X1,X2): #X1 16*16*128 X2 8*8*256
        X1=self.Unit1(X1)
        X=X1+X2#8*8*256
        out=F.softmax(self.L(self.Unit2(X).view(X.size(0),-1)))
        return out#Batch*kinds
class classifier3(nn.Module):#only High
    def __init__(self,kinds):
        super(classifier3, self).__init__()
        self.Unit1=nn.Sequential(#64*64*32
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # nn.Linear(4*4*128,kinds)
        )#8*8*256
        self.L=nn.Linear(4 * 4 *16 , kinds)
    def forward(self,X):
        X=self.Unit1(X)
        # print(X.shape)
        out=F.softmax(self.L(X.view(X.size(0),-1)))
        return out#Batch*kinds

class ChannelChange(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(ChannelChange, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.Unit=nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(True),
        )
    def forward(self,X):
        return self.Unit(X)
class HRnet(nn.Module):
    def __init__(self,Ainchannel,Binchannel,kinds):
        super(HRnet, self).__init__()
        self.R=nn.ReLU(True)
        self.Stage0=nn.Sequential(
            nn.Conv2d(Ainchannel, 32, kernel_size=3, stride=2, padding=1),#128*128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),#64*64
            nn.BatchNorm2d(64),
            nn.ReLU(True),#64*64*64
        )
        self.Stage0Y=nn.Sequential(
            nn.Conv2d(Binchannel, 32, kernel_size=3, stride=2, padding=1),#128*128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),#64*64
            nn.BatchNorm2d(64),
            nn.ReLU(True),#64*64*64
        )
        self.Stage1=nn.Sequential(#64*64
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#64*64*256
        )
        self.Stage1Y=nn.Sequential(#64*64
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#64*64*256
        )
        self.Stage1Ex=nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),  # 64*64*256
        )
        self.Stage1ExY=nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),  # 64*64*256
        )
        self.convUnit1 = Stage1Block(256)
        self.convUnit2 = Stage1Block(256)
        self.convUnit3 = Stage1Block(256)
        self.convUnit4 = Stage1Block(256)
        self.convUnit1Y = Stage1Block(256)
        self.convUnit2Y = Stage1Block(256)
        self.convUnit3Y = Stage1Block(256)
        self.convUnit4Y = Stage1Block(256)

        self.Stage2tran=Stage2Tran()
        self.Stage2tranY = Stage2Tran()

        self.S2_0convUnit1 = basicBlock(32)
        self.S2_0convUnit2 = basicBlock(32)
        self.S2_0convUnit3 = basicBlock(32)
        self.S2_0convUnit4 = basicBlock(32)
        self.S2_1convUnit1 = basicBlock(64)
        self.S2_1convUnit2 = basicBlock(64)
        self.S2_1convUnit3 = basicBlock(64)
        self.S2_1convUnit4 = basicBlock(64)

        self.S2_0convUnit1Y = basicBlock(32)
        self.S2_0convUnit2Y = basicBlock(32)
        self.S2_0convUnit3Y = basicBlock(32)
        self.S2_0convUnit4Y = basicBlock(32)
        self.S2_1convUnit1Y = basicBlock(64)
        self.S2_1convUnit2Y = basicBlock(64)
        self.S2_1convUnit3Y = basicBlock(64)
        self.S2_1convUnit4Y = basicBlock(64)

        self.stage2fusion=Stage2Fusion()
        self.Stage3tran=Stage3Tran()
        self.stage2fusionY=Stage2Fusion()
        self.Stage3tranY=Stage3Tran()

        self.S3_0convUnit1 = basicBlock(32)
        self.S3_0convUnit2 = basicBlock(32)
        self.S3_0convUnit3 = basicBlock(32)
        self.S3_0convUnit4 = basicBlock(32)
        self.S3_1convUnit1 = basicBlock(64)
        self.S3_1convUnit2 = basicBlock(64)
        self.S3_1convUnit3 = basicBlock(64)
        self.S3_1convUnit4 = basicBlock(64)
        self.S3_2convUnit1 = basicBlock(128)
        self.S3_2convUnit2 = basicBlock(128)
        self.S3_2convUnit3 = basicBlock(128)
        self.S3_2convUnit4 = basicBlock(128)

        self.S3_0convUnit1Y = basicBlock(32)
        self.S3_0convUnit2Y = basicBlock(32)
        self.S3_0convUnit3Y = basicBlock(32)
        self.S3_0convUnit4Y = basicBlock(32)
        self.S3_1convUnit1Y = basicBlock(64)
        self.S3_1convUnit2Y = basicBlock(64)
        self.S3_1convUnit3Y = basicBlock(64)
        self.S3_1convUnit4Y = basicBlock(64)
        self.S3_2convUnit1Y = basicBlock(128)
        self.S3_2convUnit2Y = basicBlock(128)
        self.S3_2convUnit3Y = basicBlock(128)
        self.S3_2convUnit4Y = basicBlock(128)


        self.Stage3fusion=Stage3Fusion()
        self.Stage4tran=Stage4Tran()

        self.Stage3fusionY=Stage3Fusion()
        self.Stage4tranY=Stage4Tran()

        self.S4_0convUnit1=basicBlock(32)
        self.S4_0convUnit2=basicBlock(32)
        self.S4_0convUnit3=basicBlock(32)
        self.S4_0convUnit4=basicBlock(32)
        self.S4_1convUnit1=basicBlock(64)
        self.S4_1convUnit2=basicBlock(64)
        self.S4_1convUnit3=basicBlock(64)
        self.S4_1convUnit4=basicBlock(64)
        self.S4_2convUnit1=basicBlock(128)
        self.S4_2convUnit2=basicBlock(128)
        self.S4_2convUnit3=basicBlock(128)
        self.S4_2convUnit4=basicBlock(128)
        self.S4_3convUnit1=basicBlock(256)
        self.S4_3convUnit2=basicBlock(256)
        self.S4_3convUnit3=basicBlock(256)
        self.S4_3convUnit4=basicBlock(256)

        self.S4_0convUnit1Y=basicBlock(32)
        self.S4_0convUnit2Y=basicBlock(32)
        self.S4_0convUnit3Y=basicBlock(32)
        self.S4_0convUnit4Y=basicBlock(32)
        self.S4_1convUnit1Y=basicBlock(64)
        self.S4_1convUnit2Y=basicBlock(64)
        self.S4_1convUnit3Y=basicBlock(64)
        self.S4_1convUnit4Y=basicBlock(64)
        self.S4_2convUnit1Y=basicBlock(128)
        self.S4_2convUnit2Y=basicBlock(128)
        self.S4_2convUnit3Y=basicBlock(128)
        self.S4_2convUnit4Y=basicBlock(128)
        self.S4_3convUnit1Y=basicBlock(256)
        self.S4_3convUnit2Y=basicBlock(256)
        self.S4_3convUnit3Y=basicBlock(256)
        self.S4_3convUnit4Y=basicBlock(256)

        self.Stage4fusion = Stage4Fusion()
        self.Classfier1 = classifier1(kinds)
        # self.Classfier1Y = classifier1(kinds)
        self.L1 = nn.Linear(8, 32)
        self.L2 = nn.Linear(32, kinds)

    def forward(self, X,Y):#DCTx 1*256*256 DctBlock 64*64*64
        X=self.Stage0(X)
        X1=self.Stage1(X)
        X2=self.Stage1Ex(X)
        Y=self.Stage0Y(Y)
        Y1=self.Stage1Y(Y)
        Y2=self.Stage1ExY(Y)

        out=self.R(X1+X2)
        out=self.convUnit1(out)
        out = self.convUnit2(out)
        out = self.convUnit3(out)
        out = self.convUnit4(out)

        outY=self.R(Y1+Y2)
        outY=self.convUnit1Y(outY)
        outY = self.convUnit2Y(outY)
        outY = self.convUnit3Y(outY)
        outY = self.convUnit4Y(outY)
        s2_0X,s2_1X=self.Stage2tran(out)

        s2_0y, s2_1y = self.Stage2tran(outY)

        s2_0=self.S2_0convUnit1(s2_0X)
        s2_0 = self.S2_0convUnit2(s2_0)
        s2_0 = self.S2_0convUnit3(s2_0)
        s2_0 = self.S2_0convUnit4(s2_0)
        s2_0=s2_0+s2_0X

        s2_1=self.S2_1convUnit1(s2_1X)
        s2_1 = self.S2_1convUnit2(s2_1)
        s2_1 = self.S2_1convUnit3(s2_1)
        s2_1 = self.S2_1convUnit4(s2_1)
        s2_1 = s2_1 + s2_1X

        s2_0Y=self.S2_0convUnit1Y(s2_0y)
        s2_0Y = self.S2_0convUnit2Y(s2_0Y)
        s2_0Y = self.S2_0convUnit3Y(s2_0Y)
        s2_0Y = self.S2_0convUnit4Y(s2_0Y)
        s2_0Y=s2_0Y+s2_0y

        s2_1Y=self.S2_1convUnit1(s2_1y)
        s2_1Y = self.S2_1convUnit2(s2_1Y)
        s2_1Y = self.S2_1convUnit3(s2_1Y)
        s2_1Y = self.S2_1convUnit4(s2_1Y)
        s2_1Y = s2_1Y + s2_1y


        s2_0,s2_1=self.stage2fusion(s2_0,s2_1)
        s2_0Y, s2_1Y = self.stage2fusionY(s2_0Y, s2_1Y)

        s3_0X,s3_1X,s3_2X=self.Stage3tran(s2_0,s2_1)
        s3_0y, s3_1y, s3_2y = self.Stage3tranY(s2_0Y, s2_1Y)

        s3_2Add=s3_2X+s3_2y

        s3_0 = self.S3_0convUnit1(s3_0X)
        s3_0 = self.S3_0convUnit2(s3_0)
        s3_0 = self.S3_0convUnit3(s3_0)
        s3_0 = self.S3_0convUnit4(s3_0)
        s3_0=s3_0+s3_0X

        s3_1 = self.S3_1convUnit1(s3_1X)
        s3_1 = self.S3_1convUnit2(s3_1)
        s3_1 = self.S3_1convUnit3(s3_1)
        s3_1 = self.S3_1convUnit4(s3_1)
        s3_1 = s3_1 + s3_1X

        s3_2 = self.S3_2convUnit1(s3_2Add)
        s3_2 = self.S3_2convUnit2(s3_2)
        s3_2 = self.S3_2convUnit3(s3_2)
        s3_2 = self.S3_2convUnit4(s3_2)
        s3_2 = s3_2Add + s3_2

        s3_0Y = self.S3_0convUnit1(s3_0y)
        s3_0Y = self.S3_0convUnit2(s3_0Y)
        s3_0Y = self.S3_0convUnit3(s3_0Y)
        s3_0Y = self.S3_0convUnit4(s3_0Y)
        s3_0Y=s3_0Y+s3_0y

        s3_1Y = self.S3_1convUnit1(s3_1y)
        s3_1Y = self.S3_1convUnit2(s3_1Y)
        s3_1Y = self.S3_1convUnit3(s3_1Y)
        s3_1Y = self.S3_1convUnit4(s3_1Y)
        s3_1Y = s3_1Y + s3_1y

        s3_2Y = self.S3_2convUnit1(s3_2Add)
        s3_2Y = self.S3_2convUnit2(s3_2Y)
        s3_2Y = self.S3_2convUnit3(s3_2Y)
        s3_2Y = self.S3_2convUnit4(s3_2Y)
        s3_2Y=s3_2Y+s3_2Add


        s4_0,s4_1,s4_2=self.Stage3fusion(s3_0,s3_1,s3_2)
        s4_0Y, s4_1Y, s4_2Y = self.Stage3fusion(s3_0Y, s3_1Y, s3_2Y)


        s4_0X,s4_1X,s4_2X,s4_3X=self.Stage4tran(s4_0,s4_1,s4_2)
        s4_0y, s4_1y, s4_2y, s4_3y = self.Stage4tran(s4_0Y, s4_1Y, s4_2Y)

        s4_3Add=s4_3X+s4_3y

        s4_0=self.S4_0convUnit1(s4_0X)
        s4_0=self.S4_0convUnit2(s4_0)
        s4_0=self.S4_0convUnit3(s4_0)
        s4_0=self.S4_0convUnit4(s4_0)
        s4_0=s4_0+s4_0X

        s4_1=self.S4_1convUnit1(s4_1X)
        s4_1=self.S4_1convUnit2(s4_1)
        s4_1=self.S4_1convUnit3(s4_1)
        s4_1=self.S4_1convUnit4(s4_1)
        s4_1 = s4_1 + s4_1X

        s4_2=self.S4_2convUnit1(s4_2X)
        s4_2=self.S4_2convUnit2(s4_2)
        s4_2=self.S4_2convUnit3(s4_2)
        s4_2=self.S4_2convUnit4(s4_2)
        s4_2 = s4_2 + s4_2X

        s4_3=self.S4_3convUnit1(s4_3Add)
        s4_3=self.S4_3convUnit2(s4_3)
        s4_3=self.S4_3convUnit3(s4_3)
        s4_3=self.S4_3convUnit4(s4_3)
        s4_3=s4_3+s4_3Add

        s4_0Y=self.S4_0convUnit1(s4_0y)
        s4_0Y=self.S4_0convUnit2(s4_0Y)
        s4_0Y=self.S4_0convUnit3(s4_0Y)
        s4_0Y=self.S4_0convUnit4(s4_0Y)
        s4_0Y=s4_0Y+s4_0y

        s4_1Y=self.S4_1convUnit1(s4_1Y)
        s4_1Y=self.S4_1convUnit2(s4_1Y)
        s4_1Y=self.S4_1convUnit3(s4_1Y)
        s4_1Y=self.S4_1convUnit4(s4_1Y)
        s4_1Y = s4_1Y + s4_1y

        s4_2Y=self.S4_2convUnit1(s4_2Y)
        s4_2Y=self.S4_2convUnit2(s4_2Y)
        s4_2Y=self.S4_2convUnit3(s4_2Y)
        s4_2Y=self.S4_2convUnit4(s4_2Y)
        s4_2Y = s4_2Y + s4_2y

        s4_3Y=self.S4_3convUnit1(s4_3Add)
        s4_3Y=self.S4_3convUnit2(s4_3Y)
        s4_3Y=self.S4_3convUnit3(s4_3Y)
        s4_3Y=self.S4_3convUnit4(s4_3Y)
        s4_3Y = s4_3Y + s4_3Add

        out=self.Classfier1(s4_0+s4_0Y,s4_1+s4_1Y,s4_2+s4_2Y,s4_3+s4_3Y)
        # out=out.view(out.size(0),-1)
        # out=F.softmax(self.L2(self.L1(out)))
        return out,s2_0X,s2_0y

# class VHRnet(nn.Module):
#     def __init__(self,inchannel,kinds):#默认256大小
#         super(VHRnet, self).__init__()
#         self.HRnet=HRnet(inchannel=inchannel,kinds=kinds)
#         self.Unet=UNet.UNet(n_channels=1,n_classes=2)
#         self.Vit=VIT.ViT(in_channels=2,img_size=256,depth=6,n_classes=2)
#     def forward(self):
#         return 0
# class doubleInHRnet(nn.Module):
#     def __init__(self,Achannel,Bchannel,kinds):
#         super(doubleInHRnet, self).__init__()
#         self.Achannel=Achannel
#         self.Bchannel=Bchannel
#         self.Anet=HRnet(inchannel=self.Achannel,kinds=kinds)
#         self.Bnet=HRnet(inchannel=self.Bchannel,kinds=kinds)
#     def forward=

if __name__=="__main__":
    net=HRnet(inchannel=1,kinds=2)
    print(net)