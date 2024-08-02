import copy
import shutil

import pandas as pd
import numpy
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import csv
import SimpleITK as sitk
import random

root='../../DataSetsV2.0'
savepath='../../Dataset-detech-liver'
path = '../DataSetsV2.0.xlsx'
labels_dict = {}
data = pd.read_excel(path)
sample_cout = np.zeros(2)
for i in range(len(data['姓名'].values)):
    for j in range(2):
        if data['是否肝脑0无1有'].values[i] == j:
            labels_dict[str(data['姓名'].values[i])] = j  # 0无1有
            sample_cout[j] += 1
print(labels_dict)
for key in labels_dict:
    if labels_dict[key]==0:
        if random.random()<0.2:#纳入测试集
            film=os.listdir(os.path.join(root,key,'肝'))#film是文件名
            for i in range(len(film)):
                shutil.copy(os.path.join(os.path.join(root,key,'肝'),film[i]),os.path.join(savepath,'val/good',key+film[i]))
        else:#训练集
            film=os.listdir(os.path.join(root,key,'肝'))
            for i in range(len(film)):
                shutil.copy(os.path.join(os.path.join(root,key,'肝'),film[i]),os.path.join(savepath,'train',key+film[i]))
    else: #HE的
        film = os.listdir(os.path.join(root, key, '肝'))
        for i in range(len(film)):
            shutil.copy(os.path.join(os.path.join(root,key,'肝'),film[i]),os.path.join(savepath,'val/HE',key+film[i]))









