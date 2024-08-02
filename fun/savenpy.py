import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import yaml
import os
import numpy as np
import csv
import model.UNet as U

def savenpy(cfg):
    path = cfg['metadata']
    data = pd.read_excel(path)
    # print('sample count%d' % len(data[cfg['sampleinfo']].values))
    for i in range(len(data[cfg['sampleinfo']].values)):
        savepath='../../savenpy/'+data[cfg['sampleinfo']].values[i]
    net=U.Unet(inchannel=1,outchannel=1)


if __name__ == "__main__":
    with open('../HE_cfg.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        print('Read yaml Success')
        f.close()
    savenpy(cfg)