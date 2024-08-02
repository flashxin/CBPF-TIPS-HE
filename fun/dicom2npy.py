import pydicom
import numpy as np
import imageio  # 转换成图像
import os
import yaml
import SimpleITK as sitk
def dcm2npy(cfg):
    # with open(cfgpath, 'r', encoding='utf-8') as f:
    #     cfg=yaml.load(f.read(),Loader=yaml.FullLoader)
    #     print('Read yaml Success')
    #     f.close()
    patients = os.listdir(cfg['dataset'])
    print('find patient Success Total: %d' % len(patients))
    os.mkdir(cfg['datanpy'])
    for patient in patients:
        if cfg['datasetmidpath'] != '':
            midpath=os.path.join(cfg['dataset'],patient,cfg['datasetmidpath'])
        else:
            midpath=patient
        # print(midpath)
        for kinds in range(len(cfg['datasetendpath'])):
            endpath=os.path.join(midpath,cfg['datasetendpath'][kinds])
            # print(endpath)
            imgs=os.listdir(endpath)
            for i in range(len(imgs)):
                # img=pydicom.dcmread(os.path.join(endpath,imgs[i])).pixel_array
                dicom=sitk.ReadImage(os.path.join(endpath,imgs[i]))
                img = np.squeeze(sitk.GetArrayFromImage(dicom))
                # img = np.array(img, dtype=float)
                # print(img.shape)
                # print(np.max(img))
                if i==0:
                    imgsize=img.shape
                    # print(imgsize)
                    npy=np.zeros([len(imgs),imgsize[0],imgsize[1]])
                else:
                    if imgsize != img.shape:
                        print('ERROR,size not match %s' % patient)
                        exit(0)
                npy[i,:]=img
            # print(os.path.join(cfg['datanpy'],os.path.basename(patient),cfg['datasetendpath'][kinds]))
            np.save(os.path.join(cfg['datanpy'],os.path.basename(patient)+cfg['datasetendpath'][kinds]+'.npy'),npy)


def dcm2npyV2(cfg,patients):
    print('find patient Success Total: %d' % len(patients))
    if not os.path.exists(cfg['datanpy']):
        os.mkdir(cfg['datanpy'])
    for patient in patients:
        if cfg['datasetmidpath'] != '':
            midpath=os.path.join(cfg['dataset'],patient,cfg['datasetmidpath'])
        else:
            midpath=os.path.join(cfg['dataset'],patient)
        # print(midpath)
        for kinds in range(len(cfg['datasetendpath'])):
            endpath=os.path.join(midpath,cfg['datasetendpath'][kinds])
            # print(endpath)
            imgs=os.listdir(endpath)
            for i in range(len(imgs)):
                # img=pydicom.dcmread(os.path.join(endpath,imgs[i])).pixel_array
                dicom=sitk.ReadImage(os.path.join(endpath,imgs[i]))
                img = np.squeeze(sitk.GetArrayFromImage(dicom))
                # img = np.array(img, dtype=float)
                # print(img.shape)
                # print(np.max(img))
                if i==0:
                    imgsize=img.shape
                    # print(imgsize)
                    npy=np.zeros([len(imgs),imgsize[0],imgsize[1]])
                else:
                    if imgsize != img.shape:
                        print('ERROR,size not match %s' % patient)
                        exit(0)
                npy[i,:]=img
            # print(os.path.join(cfg['datanpy'],os.path.basename(patient),cfg['datasetendpath'][kinds]))
            np.save(os.path.join(cfg['datanpy'],os.path.basename(patient)+cfg['datasetendpath'][kinds]+'.npy'),npy)






# if __name__=="__main__":
