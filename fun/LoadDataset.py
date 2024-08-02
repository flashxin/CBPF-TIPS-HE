import copy

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
# val_data_transform = transforms.Compose([
#     # transforms.ToTensor(),
#     # transforms.Resize([224,224]),
#     transforms.RandomRotation(45, expand=False),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomAdjustSharpness(0.8),
#     transforms.RandomAffine(45)
# ])
def gamma_transformation(input_image, c, gamma):
    '''
    伽马变换
    :param input_image: 原图像
    :param c: 伽马变换超参数
    :param gamma: 伽马值
    :return: 伽马变换后的图像
    '''
    img_norm = input_image / 255.0  # 注意255.0得采用浮点数
    img_gamma = c * np.power(img_norm, gamma) * 255.0
    img_gamma[img_gamma > 255] = 255
    img_gamma[img_gamma < 0] = 0
    # img_gamma = img_gamma.astype(np.uint8)

    return img_gamma
def data_transform(x):
    if random.random()<0.4:
        x=x.T
    if random.random()<0.45:
        x=np.flipud(x)
    if random.random()<0.45:
        x=np.fliplr(x)

    max=np.max(x)
    min=np.min(x)

    if random.random()<0.45:
        x=np.power((x-min)/(0.0001+(max-min)),random.random()*5)*x
    return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,cfg,mode,samplelist,labeldir,blacklist):
        super(MyDataset, self).__init__()
        self.cfg=cfg
        self.labeldir=labeldir
        self.black=blacklist
        # print(blacklist)
        self.changesample=list(filter(lambda k: labeldir[k] == 1, labeldir.keys()))
        print(len(self.changesample))
        for i in self.changesample:
            # print(i)
            if i in self.black:
                self.changesample.remove(i)
        print(len(self.changesample))
        self.samplelist=samplelist
        print('Load dataseting......')
        root=cfg['datanpy']
        self.samplelist=samplelist
        self.moudle=mode
        if self.moudle=='train':
            print('_________________')
            print('Train moudles now')
            print('_________________')
        elif self.moudle=='val':
            print('_________________')
            print('Val moudles now')
            print('_________________')
        else:
            print('ERROR MODE!PLEASE CHECK config')
            exit(0)
        print('Load Dataset Success!')
        if self.moudle=='train':
            print('There are %d samples at %s dataset' % (len(samplelist)*cfg['data_extern_train'], self.moudle))
        else:
            print('There are %d samples at %s dataset'% (len(samplelist)*cfg['data_extern_val'],self.moudle))
        print('data shape [ datasetendpath, slice, H, W]')
    def __len__(self):
        if self.moudle=='train':
            return len(self.samplelist)*self.cfg['data_extern_train']
        return len(self.samplelist)*self.cfg['data_extern_val']
    def __getitem__(self, item):
        if self.moudle=='train':
            name = self.samplelist[item%len(self.samplelist)]
            #add random change
            if self.labeldir[name]==0 and random.random()<0.2:
                name=self.changesample[random.randrange(0,len(self.changesample))]
        else:
            name=self.samplelist[item%len(self.samplelist)]


        # print(name)
        # npys=np.zeros(len(self.cfg['datasetendpath']))
        # npys=[[] for i in range(len(self.cfg['datasetendpath']))]
        npysA=np.zeros([self.cfg['slicenumber'][0],self.cfg['imagesize'],self.cfg['imagesize']])
        npysB = np.zeros([self.cfg['slicenumber'][1], self.cfg['imagesize'], self.cfg['imagesize']])
        for i in range(len(self.cfg['datasetendpath'])):
            img=np.load(os.path.join(self.cfg['datanpy'],name+self.cfg['datasetendpath'][i]+'.npy'))
            # img=img[:,::2,::2]
            choice=random.sample(range(0,img.shape[0]),int(self.cfg['slicenumber'][i]))
            # print(img.shape)
            img=img.take([choice],axis=0)
            img=np.squeeze(img,axis=0)
            # print(img.shape)
            # img.resize([self.cfg['slicenumber'],self.cfg['imagesize'],self.cfg['imagesize']])
            # print(img.shape)
            # print(img.shape)
            output=np.zeros([img.shape[0],self.cfg['imagesize'],self.cfg['imagesize']])
            for k in range(img.shape[0]):
            #图片增强
                # if self.moudle=='val':
                output[k,:]=resize_img_keep_ratio(img[k,:],[self.cfg['imagesize'],self.cfg['imagesize']])
                # else:
                # output[k, :] = data_transform(resize_img_keep_ratio(img[k, :], [self.cfg['imagesize'], self.cfg['imagesize']]))
            # print(i)
            if i==0:
                npysA=output
            elif i==1:
                npysB=output
        # npys=np.array(npys)
        # npys.resize([len(self.cfg['datasetendpath'])])
        npysA=np.array(npysA)
        npysB = np.array(npysB)
        # npysA=torch.tensor(npysA)
        # npysB = torch.tensor(npysB)
        # print(np.max(npysA))
        # print(np.max(npysB))
        # if self.moudle=='train':
        #     # npys=torch.tensor(npys)
        #     # npys=train_data_transform(npys)
        # if self.moudle=='val':
        #     npys=val_data_transform(npys)
        # print(npys.shape)
        return name,npysA,npysB

def resize_img_keep_ratio(img,target_size):
    # img = cv2.imread(img_name) # 读取图片
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0))
    # img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    # print(img_new.shape)
    return img_new

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,cfg,mode,samplelist,bioinfo):
        super(TextDataset, self).__init__()
        self.cfg=cfg
        self.samplelist=samplelist
        print('Load dataseting......')
        root=cfg['datanpy']
        self.samplelist=samplelist
        self.moudle=mode
        self.bioinfo=bioinfo
        if self.moudle=='train':
            print('_________________')
            print('Train moudles now')
            print('_________________')
        elif self.moudle=='val':
            print('_________________')
            print('Val moudles now')
            print('_________________')
        else:
            print('ERROR MODE!PLEASE CHECK config')
            exit(0)
        print('Load Dataset Success!')
    def __len__(self):
        return len(self.samplelist)
    def __getitem__(self, item):
        name = self.samplelist[item]
        data=self.bioinfo[name]
        # mask=random.random()
        # if mask< self.cfg['mask_rate']:
        for i in range(len(self.cfg['bioinfo'])):
            if random.random() < self.cfg['shake_rate'] and data[i] != -1:
                data[i]=data[i]*(1+ random.uniform(-self.cfg['shake_amp'], self.cfg['shake_amp']))
                # data[i]=data[i]*(1+random.random)
            if random.random()<self.cfg['mask_rate'] and data[i]!=-1:
                data[i]=-1
        # print(data)
        # data=np.array(data)
        # print(data.shape)#51 个项目
        return name,data
class preTrainData(torch.utils.data.Dataset):
    def __init__(self,cfg,mode):
        super(preTrainData, self).__init__()
        self.mode=mode
        self.cfg=cfg
        # self.samplelist=samplelist
        with open(cfg['dataloaderpath'], 'r') as f:
            reader = csv.reader(f)
            result = list(reader)
        self.result=result
        print('Load %d sample'%len(self.result))
    def __len__(self):
        return len(self.result)

    def __getitem__(self, item):
        dicom = sitk.ReadImage(os.path.join(self.cfg['dataset'], str(self.result[item][0])))
        img = np.squeeze(sitk.GetArrayFromImage(dicom))
        label=copy.deepcopy(img)
        n_block=int(img.shape[0]*img.shape[1]*self.cfg['maskrate']/(self.cfg['blocksize']**2))
        for i in range(n_block):
            x=np.random.randint(30,img.shape[0]-self.cfg['blocksize']-30)
            y = np.random.randint(30, img.shape[1] - self.cfg['blocksize'] - 30)
            img[x:x+self.cfg['blocksize'],y:y+self.cfg['blocksize']]=-1

        img = img[::2,::2]
        label = label[::2,::2]
        img=np.expand_dims(img,axis=0)
        label=np.expand_dims(label,axis=0)

        return img,label
train_data_transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize([224,224]),
    transforms.RandomRotation(45, expand=False),
    transforms.RandomHorizontalFlip(),
    transforms.RandomErasing(p=0.5,scale=(0.02,0.33),value=0)
])
val_data_transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize([224,224]),
    # transforms.RandomRotation(45, expand=False),
    # transforms.RandomHorizontalFlip(),
])


class TrainClass(torch.utils.data.Dataset):
    def __init__(self,cfg,mode,samplelist,liverdir):
        super(TrainClass, self).__init__()
        self.mode=mode
        self.cfg=cfg
        self.samplelist=samplelist
        self.liverdir=liverdir
        print('※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※')
        print('Load Sample %d as %s dataset'%(len(self.samplelist),self.mode))
        print('※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※')
    def __len__(self):
        return len(self.samplelist)
    def __getitem__(self, item):
        # Liverpath=os.path.join(self.cfg['dataset'],self.samplelist[item])
        # filmsLiver=os.listdir(os.path.join(Liverpath,'肝'))
        # for f in filmsLiver:
        #     dicom = sitk.ReadImage(f)
        #     img = np.squeeze(sitk.GetArrayFromImage(dicom))
        img=self.liverdir[self.samplelist[item]]
        if self.mode=='val':
            return self.samplelist[item], val_data_transform(img)
        return self.samplelist[item],train_data_transform(img)
if __name__ == "__main__":
    with open('preTrainliver.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        print(len(result))