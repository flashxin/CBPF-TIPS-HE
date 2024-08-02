import imageio  # 转换成图像
import os
import SimpleITK
import pydicom
import numpy as np
def get_file(root_path, all_files=[]):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件夹，获取其path
    '''
    files = os.listdir(root_path)
    for file in files:
        # if not os.path.isdir(root_path + '/' + file):  # not a dir
        #     if os.path.splitext(file)[1] == '':
        #         all_files.append(root_path + '/' + file)  # os.path.basename(file)
        # else:  # is a dir
        #     get_file((root_path + '/' + file), all_files)
        if  os.path.isdir(root_path + '/' + file):
            all_files.append(root_path + '/' + file)
    return all_files
def get_all(root_path, all_files=[]):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件夹，获取其path
    '''
    files = os.listdir(root_path)
    for file in files:
        if not os.path.isdir(root_path + '/' + file):  # not a dir
            if os.path.splitext(file)[1] == '':
                all_files.append(root_path + '/' + file)  # os.path.basename(file)
        else:  # is a dir
            get_all((root_path + '/' + file), all_files)
        if  os.path.isdir(root_path + '/' + file):
            all_files.append(root_path + '/' + file)
    return all_files

def dcm_to_image(dcmfile,imgfile):
    print(dcmfile)
    seq_of_slice = []
    filenames=os.listdir(dcmfile)
    for f in filenames:
        # read=SimpleITK.ImageSeriesReader()
        # dcmfile=read.GetGDCMSeriesFileNames(f)
        # read.SetFileNames(dcmfile)
        # img = read.Execute()  # 读取dcm的SliceLocation
        # imgfiles=os.listdir(f)
        meta=pydicom.dcmread(dcmfile+'/'+os.path.basename(f))
        seq_of_slice.append(meta.SliceLocation)
        # print(dcmfile+'/'+os.path.basename(f))
    seq_of_slice.sort()
    print(seq_of_slice)
    for f in filenames:
        meta=pydicom.dcmread(dcmfile+'/'+os.path.basename(f))
        locals=meta.SliceLocation
        for j in range(len(seq_of_slice)):
            if seq_of_slice[j]==locals:
                print(os.path.exists(dcmfile+'/'+os.path.basename(f)))
                try:
                    os.rename(dcmfile+'/'+os.path.basename(f),os.path.join(dcmfile,str(j)+'.DCM'))
                except:
                    print('e')
    # everyfilms=get_all(dcmfile)
    # for f in everyfilms:
    #     father_path=os.path.abspath(os.path.dirname(f)+os.path.sep+".")#读取上一层路径
    #     img=pydicom.dcmread(f)
    #     img_array=img.pixel_array
    #     arr_temp = img_array.reshape(-1)
    #     max_val = max(arr_temp)
    #     min_val = min(arr_temp)
    #     # 像素值归一化
    #     img_array = (img_array - min_val) / (max_val - min_val)
    #     img_tmp = (img_array * 255).astype(np.uint8)
    #
    #     fname = os.path.basename(f).replace('IMG', '_')  # 去掉dcm的后缀名
    #     fname=str(os.path.basename(father_path)+fname+'.png')
    #     imageio.imwrite(str(imgfile+'/'+fname), img_tmp) # 保存图像


if __name__ == '__main__':
    filepath = '../全期'
    imgfile = '../lajidui/'
    dcm_to_image(filepath,imgfile)
    # meta = pydicom.dcmread("E:\yuepian\尘肺病阅片系统\Gong Xing Long\\1.3.46.670589.33.1.63772569598582736600001.4684186496472500514.dcm")
    #print(meta.elements("FileMetaInformationVersion"))