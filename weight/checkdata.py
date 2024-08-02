import pandas as pd
import yaml
import os
import numpy as np
blackdata=pd.read_excel('../metadata.xlsx')
blacklist=[]
for i in range(len(blackdata['姓名'].values)):
    blacklist.append(str(blackdata['姓名'].values[i]))
# print(blacklist)
normallist=[]
a=np.load('3DRes_动静脉联合5倍扩充3D_name_1_fold.npy')
for i in range(len(a)):
    normallist.append(a[i])
a=np.load('3DRes_动静脉联合5倍扩充3D_name_2_fold.npy')
for i in range(len(a)):
    normallist.append(a[i])
a=np.load('3DRes_动静脉联合5倍扩充3D_name_3_fold.npy')
for i in range(len(a)):
    normallist.append(a[i])
a=np.load('3DRes_动静脉联合5倍扩充3D_name_4_fold.npy')
for i in range(len(a)):
    normallist.append(a[i])
# print(normallist)
maybenormallist=[]
maybeerror=[]
for i in range(len(blacklist)):
    if blacklist[i] in normallist:
        maybenormallist.append(blacklist[i])
    else:
        maybeerror.append(blacklist[i])

# for i in range(len(dellist)):
#     print(dellist[i])
for i in range(len(maybeerror)):
    print(maybeerror[i])