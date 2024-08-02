import torch
from torch import nn
import math
class MultiAtt(nn.Module):
    def __init__(self,embsize,size=3):
        super(MultiAtt,self).__init__()
        self.WQ = nn.Parameter(torch.zeros(size, embsize), requires_grad=True)
        self.WK = nn.Parameter(torch.zeros(size, embsize), requires_grad=True)
        self.WV = nn.Parameter(torch.zeros(size, embsize), requires_grad=True)
        self.Soft=nn.Softmax()
        self.embsize=embsize
        self.size=size
        # self.headnum=headnum
        self.exten=nn.Linear(embsize,embsize*size)
        self.shrink = nn.Linear(size**2, embsize)
    def forward(self,x):
        # print(x.shape)
        # x = self.exten(x)
        # print(x.shape)
        mat=self.exten(x).view(x.size(0),self.embsize,self.size)
        # print(x.shape)
        Q  =torch.matmul(self.WQ,mat)
        K = torch.matmul(self.WQ, mat)
        V = torch.matmul(self.WQ, mat)
        #att_size  size[0], size[1]
        att=torch.matmul(self.Soft(torch.matmul(Q,K.permute(0,2,1))/self.embsize**(1/2)),V).view(x.size(0),-1)
        # print(att.shape)
        out=self.shrink(att)
        # out=x*out
        return out

#Linear U-net
class TextEncoder(nn.Module):
    def __init__(self,inputsize,outputsize,mode='train'):
        super(TextEncoder, self).__init__()
        self.inputsize=inputsize
        self.mode=mode
        self.outputsize=outputsize
        self.att=MultiAtt(embsize=inputsize,size=3)
        self.test=nn.Linear(inputsize,outputsize)
        self.fc1=nn.Sequential(
            nn.Linear(in_features=inputsize,out_features=inputsize*2),
            nn.Linear(inputsize*2,inputsize*2),
            nn.Linear(inputsize*2, inputsize),

        )
        self.fc2=nn.Sequential(
            nn.Linear(in_features=inputsize,out_features=inputsize*2),
            nn.Linear(inputsize*2,inputsize*2),
            nn.Linear(inputsize*2, inputsize),

        )
        self.fc3=nn.Sequential(
            nn.Linear(in_features=inputsize,out_features=inputsize*2),
            nn.Linear(inputsize*2,inputsize*2),
            nn.Linear(inputsize*2, inputsize),
        )
        self.fc4=nn.Sequential(
            nn.Linear(in_features=inputsize,out_features=inputsize*2),
            nn.Linear(inputsize*2,inputsize*2),
            nn.Linear(inputsize*2, inputsize),
        )
        # self.Encoder.add_module('att1',MultiAtt(embsize=13))
    def forward(self,x):
        out1=x+self.fc1(x)
        out2=out1+self.fc2(x)
        # Mid=self.att(out2)
        out3=out2+self.fc3(out2)
        out4=out3+self.fc4(out3+out1)
        return out4


