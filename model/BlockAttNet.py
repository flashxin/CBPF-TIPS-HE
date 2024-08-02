from torch import nn
import math

class BANet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x