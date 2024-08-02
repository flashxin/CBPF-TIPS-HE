import pandas as pd
import yaml
import os
import numpy as np


def Getlable(cfg):
    # with open(cfgpath, 'r', encoding='utf-8') as f:
    #     cfg=yaml.load(f.read(),Loader=yaml.FullLoader)
    #     print('Read yaml Success')
    #     f.close()
    # print(cfg)
    path = cfg['metadata']
    baseinfo = cfg['baseinfo']
    bioinfo = cfg['bioinfo']
    labels_dict = {}
    baseInfo_dict = {}
    bio_dict = {}
    edema_dict = {}
    data = pd.read_excel(path)
    sample_cout = np.zeros([cfg['kind']])
    # print('sample count%d' % len(data[cfg['sampleinfo']].values))
    for i in range(len(data[cfg['sampleinfo']].values)):
        for j in range(cfg['kind']):
            if data[cfg['lableinfo']].values[i] == j:
                labels_dict[str(data[cfg['sampleinfo']].values[i])] = j  # 0无1有
                sample_cout[j] += 1
        # print(len(baseinfo))
        baseinfo_list = np.zeros(len(baseinfo))
        for j in range(len(baseinfo)):
            baseinfo_list[j]=np.array(data[str(baseinfo[j])].values[i])
        baseInfo_dict[str(data[cfg['sampleinfo']].values[i])] = baseinfo_list
        # print(baseinfo_list.size)
        bio_list = np.zeros(len(bioinfo))
        for k in range(len(bioinfo)):
            bio_list[k]=np.array(data[str(bioinfo[k])].values[i])
        bio_dict[str(data[cfg['sampleinfo']].values[i])] = bio_list
        edema_dict[str(data[cfg['sampleinfo']].values[i])]=data[str(cfg['edemainfo'])].values[i]
        # empty data write as nan
    print('The sample total count is %d' % np.sum(sample_cout))
    for i in range(cfg['kind']):
        print('%s sample have %d items' % (cfg['kindname'][i], sample_cout[i]))
    # print('Demo:')
    # print(baseInfo_dict['白雷'],baseInfo_dict['李霞'])
    # print(bio_dict['白雷'], bio_dict['李霞'])
    return [labels_dict, baseInfo_dict, bio_dict,edema_dict], sample_cout
# def GetEdema(cfg,sample_cout):
#     path=cfg['edema']
#     labels_dict = {}
#     data = pd.read_excel(path)
def GetHElable(cfg):
    # with open(cfgpath, 'r', encoding='utf-8') as f:
    #     cfg=yaml.load(f.read(),Loader=yaml.FullLoader)
    #     print('Read yaml Success')
    #     f.close()
    # print(cfg)
    path = cfg['metadata']
    labels_dict = {}
    data = pd.read_excel(path)
    sample_cout = np.zeros([cfg['kind']])
    # print('sample count%d' % len(data[cfg['sampleinfo']].values))
    for i in range(len(data[cfg['sampleinfo']].values)):
        for j in range(cfg['kind']):
            if data[cfg['lableinfo']].values[i] == j:
                labels_dict[str(data[cfg['sampleinfo']].values[i])] = j  # 0无1有
                sample_cout[j] += 1
        # print(len(baseinfo))
    print('The sample total count is %d' % np.sum(sample_cout))
    for i in range(cfg['kind']):
        print('%s sample have %d items' % (cfg['kindname'][i], sample_cout[i]))
    # print('Demo:')
    # print(baseInfo_dict['白雷'],baseInfo_dict['李霞'])
    # print(bio_dict['白雷'], bio_dict['李霞'])
    return labels_dict, sample_cout
if __name__ == "__main__":
    Getlable('../cfg.yaml')
