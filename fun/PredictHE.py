import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor,ceil

class SPP(nn.Module):
    def __init__(self,level_num,pool_type='max_pool'):
        super(SPP, self).__init__()
        self.level_num=level_num
        self.pool_type=pool_type

    def forward(self,x):
        N,C,H,W=x.size()
        for i in range(self.level_num):
            level = i+1
            print('第',level,'次计算卷积核')
            kernel_size=(ceil(H/level),ceil(W/level))
            stride=(ceil(H/level),ceil(W/level))
            padding=(floor((kernel_size[0]*level-H+1)/2),floor((kernel_size[1]*level-W+1)/2))
            if self.pool_type == 'max_pool':
                feature_map=(torch.nn.functional.max_pool2d(kernel_size=kernel_size,stride=stride,padding=padding,input=x)).view(N,-1)
            else:
                feature_map=(torch.nn.functional.avg_pool2d(kernel_size=kernel_size,stride=stride,padding=padding,input=x)).view(N,-1)

            if i == 0:
                result=feature_map
            else:
                result=torch.cat((result,feature_map),1)
                print(result.size())
        return result
if __name__ == "__main__":
    a=torch.rand([1,1,45,567])
    # print(a)
    spp=SPP(4)
    # print(spp(a))
    print(spp(a).shape)
