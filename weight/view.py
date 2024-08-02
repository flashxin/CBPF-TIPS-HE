import numpy as np
a=np.load('单腰肝5抽样4-1_name_0_fold.npy')
b=np.load('单腰肝5抽样4-1_pred_0_fold.npy')
v=np.load('单腰肝5抽样4-1_label_0_fold.npy')
# n=(len(a)//5)
# for i in range(n):
#     # print(a)
#     print(a[i],v[i],b[i],b[i+n],b[i+n*2],b[i+n*3],b[i+n*4])
for i in range(len(a)):
    print(a[[i]])
    print(v[i])
    print(b[i])

import numpy
import torch.utils.data as Data #批处理模块
import torch
from torch.autograd.gradcheck import gradcheck
import torchvision.datasets as dset
import copy
from torchvision import transforms
from torch import nn
import math
import torch.nn.functional as F
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import tqdm
import numpy as np
import matplotlib.pyplot as plt

def roc_draw(predict, ground_truth):
    nums = len(predict)
    x, y = 1, 1
    index = np.argsort(predict)
    ground = ground_truth[index]

    x_step = 1.0 / (nums - sum(ground_truth))  # 负样本步长
    y_step = 1. / sum(ground_truth)

    res_x = []
    res_y = []

    for i in range(nums):
        if ground[i] == 1:
            y -= y_step
        else:
            x -= x_step

        res_x.append(x)
        res_y.append(y)
    return res_x, res_y


from sklearn.metrics import roc_curve
fpr, tpr, thersholds = roc_curve(v, b, pos_label=1)
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.4f})'.format(roc_auc), lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.show()
precision, recall, thresholds = precision_recall_curve(v, b)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot(recall, precision)
plt.xlabel('R')
plt.ylabel('P')  # 可以使用中文，但需要导入一些库即字体
plt.show()
