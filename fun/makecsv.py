import pandas as pd
import yaml
import os
import numpy as np
import csv


# for i in range(len(listA)):
#     f = open("filename.csv", 'a', newline='')
#     writer = csv.writer(f)
#     writer.writerow(listA[i])
#     f.close()
def ImagePath(cfg):
    samplepath='../DataSetsV2.0.xlsx'
    data = pd.read_excel(samplepath)
    f = open("preTrainYao.csv", 'a', newline='')
    writer = csv.writer(f)
    for i in range(len(data[cfg['sampleinfo']].values)):
        midpath=str(data[cfg['sampleinfo']].values[i])
        films=os.listdir(os.path.join('../../DataSetsV2.0/',midpath,'腰椎'))
        # print(films[0])
        for j in range(len(films)):
            # print(films[j])
            writer.writerow([midpath+'/腰椎/'+films[j]])#变列表
    f.close()
if __name__ == "__main__":
    with open('../HE_cfg.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        print('Read yaml Success')
        f.close()
    ImagePath(cfg)