import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import cv2
import numpy as np
import yaml
from einops import rearrange, reduce, repeat
import fun.SPP
import fun.makeLable as mklab
import matplotlib.pyplot as plt
import fun.dicom2npy as d2n
import fun.K_fold as kf
import model.UNet as U
import fun.LoadDataset as Dload
import torch.utils.data as Data  # 批处理模块
import model.Resnet as md
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import math
from torchvision import transforms
import model.triDres as mc
import sys
import SimpleITK as sitk

def Multi2Map(cfg,net,samplist,device):
    with torch.no_grad():
        spp2=fun.SPP.SPP(2)
        spp3=fun.SPP.SPP(5)
        Livermapdir={}
        for i in range(len(samplist)):
            people=os.path.join(cfg['dataset'],samplist[i])#读取每一个病例
            Liverfilms=os.listdir(os.path.join(people,'肝'))
            Yaofilms=os.listdir(os.path.join(people,'腰椎'))
            livermap=torch.zeros([len(Liverfilms),2560])
            n=0
            for f in Liverfilms:
                dicom = sitk.ReadImage(os.path.join(people,'肝',f))
                img = np.squeeze(sitk.GetArrayFromImage(dicom))
                img = img[::2, ::2]
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=0)#变成4维度
                img=torch.tensor(img).type(torch.FloatTensor).to(device)
                _,x1,x2,x3,x4,x5=net(img)
                # x5=x5[:,::2,::2,::2]#降采样
                out=spp2(x5).view(-1)#x5=1, 512, 32, 32  out 2560
                # print(out.shape)
                livermap[n,:]=out
                n+=1
            livermap=rearrange(livermap,'h (b d a) -> a b h d',a=1,b=256)
            # print(livermap.shape)
            livermap=spp3(livermap)#23296
            # print(livermap.shape)
            livermap=livermap.resize_([112,112]).unsqueeze(dim=0)
            # print(livermap.shape)
            # livermap=F.interpolate(livermap, size=[224,224], mode="bilinear")
            Livermapdir[samplist[i]]=livermap
            # print(Livermapdir[samplist[i]])
    return Livermapdir#1*112*112