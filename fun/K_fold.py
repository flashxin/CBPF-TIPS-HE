import numpy as np
import random
from random import sample
import copy

def K_fold(cfg,samplecount,lableDir):
    one_fold=int(np.sum(samplecount)//cfg['Kfold'])
    keys = list(lableDir.keys())
    trainsets=[0 for i in range(cfg['Kfold'])]
    valsets=[0 for i in range(cfg['Kfold'])]
    restset=copy.deepcopy(keys)
    for i in range(cfg['Kfold']-1):
        getsample  = random.sample(range(0,len(restset)),one_fold)
        # print(getsample)
        valset = []
        for j in range(len(getsample)):
            valset.append(restset[getsample[j]])
        restset=list(set(restset)-set(valset))
        trainset=list(set(keys)-set(valset))
        trainsets[i]=np.array(trainset)
        valsets[i]=np.array(valset)
        # print(valset)
    valsets[cfg['Kfold']-1]=np.array(list(restset))
    trainsets[cfg['Kfold']-1]=np.array(list(set(keys)-set(restset)))
    trainsets=np.array(trainsets)
    valsets=np.array(valsets)
    # print(trainsets[1][2])
    # print(valsets[4][3])
    print('_______________WARRING!!!_________________________________')
    print('transets[i] means K fold  train datesets,dtype is np.array')
    print('transets[i][j] means  j-th sample in i-th fold')
    print('__________________________________________________________')
    print('dataset have been slice to %d blocks,one of them as val ' % cfg['Kfold'])
    print('we recommend kfold = 5 or 10')
    return trainsets,valsets

