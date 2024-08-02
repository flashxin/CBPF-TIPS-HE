import os.path
import cv2
import numpy as np
import yaml
import fun.makeLable as mklab
import matplotlib.pyplot as plt
import fun.dicom2npy as d2n
import fun.K_fold as kf
import fun.LoadDataset as Dload
import torch.utils.data as Data  # 批处理模块
import model.Resnet as md
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import math
from torchvision import transforms
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from thop import profile
import model.triDres as mc
import sys
import cv2.cv2
import fun.lossfun as Loss

map=False

def grad_cam(model, input_tensor, target_class=None):
    # 设置模型为evaluation模式
    model.eval()

    # 计算梯度
    input_tensor.requires_grad_(True)

    # 前向传播
    output = model(input_tensor)

    if target_class is None:
        target_class = torch.argmax(output)

    # 反向传播，计算梯度
    model.zero_grad()
    output[0, target_class].backward()

    # 获取目标层的梯度
    gradients = input_tensor.grad.data

    # 获取目标层的权重
    activations = model.features[-1].forward(input_tensor)#指定层
    weight = torch.mean(gradients, axis=(2, 3), keepdim=True)
    cam = torch.sum(weight * activations, axis=1, keepdim=True)
    cam = torch.relu(cam)

    # 将cam映射到原始图像上
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))

    return cam[0, 0, :, :].detach().numpy()
if __name__ == "__main__":
    with open('cfgV3.0.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        print('Read yaml Success')
        f.close()
    if os.path.exists(cfg['datanpy']):
        print('npy data have been creat!')
    else:
        d2n.dcm2npy(cfg)
        print('.npy Save Success')
    if cfg['GPUmode'] == 'custom':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['GPUseq']
    print('GPUmode is %s' % cfg['GPUmode'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:   %s' % device)
    net = md.doubleAttResModel(Achannel=cfg['slicenumber'][0], Bchannel=cfg['slicenumber'][1], kinds=cfg['kind'])
    net.to(device)
    net.load_state_dict(torch.load('weight/单腰肝5抽样4-1_3_fold.pt'))
    # print(device == 'cuda' and torch.cuda.device_count() > 1 and cfg['GPUpara'] == True)
    if (torch.cuda.device_count() > 1) and (cfg['GPUpara'] == True):
        net = torch.nn.DataParallel(net)
        print('Open GPUs Parallel eval')

    name = np.load('weight/单腰肝5抽样4-1_name_3_fold.npy')
    names = np.unique(name)
    workers = cfg['workers']
    BATCH_SIZE = cfg['batchsize']

    lable, samplecount = mklab.GetHElable(cfg)

    val_dataset = Dload.MyDataset(cfg, 'val', names,lable,[])
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=1,
                                 drop_last=False)
    all_name = []
    all_label = []
    all_pred = []
    all_prob = np.zeros([len(names) * cfg['data_extern_val'], cfg['kind']])
    num = 0

    #  出参数量和计算量
    # arrA = torch.randn([1,cfg['slicenumber'][0], cfg['imagesize'], cfg['imagesize']]).to(device)
    # arrB = torch.randn([1, cfg['slicenumber'][1], cfg['imagesize'], cfg['imagesize']]).to(device)
    # # print(image_arr.shape)
    # flops, params = profile(net, (arrA,arrB))
    # print('flops: ', flops, 'params: ', params)  # 参数量46.01M 25.01 M
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    with torch.no_grad():
        net.train()
        val_loader = tqdm(val_loader, file=sys.stdout)
        for step, (name, npysA,npysB) in enumerate(val_loader):
            # net.train()
            label = torch.zeros(len(name))
            for i in range(len(name)):
                # print(name[i])
                label[i] = lable[name[i]]
            imgA = npysA.type(torch.FloatTensor)
            imgB = npysB.type(torch.FloatTensor)
            # img = train_data_transform(img)
            label = label.type(torch.LongTensor)
            imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)
            output,A,B = net(imgA,imgB)#出图
            pred = torch.max(output, 1)[1].data.cpu().numpy()

            # 将cam映射到原始图像上

            for i in range(len(name)):
                # print(all_label)
                all_name.append(name[i])
                all_label.append(lable[name[i]])
                all_pred.append(pred[i])
                for j in range(cfg['kind']):
                    all_prob[num, j] = (nn.functional.softmax(output[i], dim=-1).data.cpu().numpy())[j]
                num += 1

            # while True:
            #     print(B.shape)
            if map == True:
                heatmap = (A.data.cpu().numpy())[0, :]  # 取第一张
                heatmap = np.maximum(heatmap, 0)  # heatmap与0比较
                heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
                heatmap /= np.max(heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
                # print(heatmap)
                # 创建自定义颜色映射
                heatmap = cv2.resize(heatmap, (cfg['imagesize'], cfg['imagesize']))  # 特征图的大小调整为与原始图像相同
                heatmap = 255-np.uint8(255 * heatmap)  # 将特征图转换为uint8格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # lower_red = np.array([0, 0, 128])
                # upper_red = np.array([0, 0, 255])
                # mask_red = cv2.inRange(heatmap, lower_red, upper_red)
                # plt.matshow(heatmap)  # 可以通过 plt.matshow 显示热力图
                # plt.axis('off')
                # plt.show()
                plt.imsave(name[0]+'hetmap.jpg', heatmap, cmap='gray')
                # 将红色部分替换为白色
                # heatmap[mask_red > 0] = [255, 255, 255]
                # print(heatmap)
                from PIL import Image
                # for i in range(cfg['slicenumber'][1]):
                for i in range(1):
                    oriimg = npysA[0,i,:].data.cpu().numpy()
                    plt.imsave(name[0]+'oriimg.jpg', oriimg, cmap='gray')
                    oriimg = cv2.imread(name[0]+'oriimg.jpg')
                    # plt.matshow(oriimg)  # 可以通过 plt.matshow 显示热力图
                    # plt.show()
                    heat_img = cv2.addWeighted(oriimg,1, heatmap, 0.5 ,0)  # 将伪彩色图与原始图片融合
                    # plt.matshow(heat_img)  # 可以通过 plt.matshow 显示热力图print(name)
                    # print(name,pred)
                    # plt.show()


    # all_label=np.array(all_label)
    # accuracy = float((all_pred == torch.tensor(all_label).data.cpu().numpy()).astype(int).sum()) / float(
    #     torch.tensor(all_label).size(0))

    for i in range(len(all_name)):
        print(all_name[i], all_label[i], all_pred[i])
    tagname = cfg['kindname']
    support_dict = classification_report(all_label, all_pred, target_names=tagname, output_dict=True)
    print(len(all_pred))
    print("accuracy:", support_dict['accuracy'])
    matrix = confusion_matrix(all_label, all_pred)
    plt.matshow(matrix, cmap='YlOrRd')
    plt.colorbar()
    plt.title("Confusion_Matrix")
    plt.show()

    # print(all_prob.shape)

    print("混淆矩阵:\n", matrix)
    print(classification_report(all_label, all_pred, target_names=tagname))
    print("#==========================3.ROC曲线_AUC值=======================================#")
    fpr, tpr, threshold = roc_curve(all_label, all_prob[:, 0], pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print("auc:", auc)
    # [0.  0.11111111 0.11111111 0.22222222 0.22222222 0.88888889 1.]
    # [0.   0.75       0.8           0.8         0.95     0.95    1.]
    # 绘制roc曲线
    plt.plot(fpr, tpr, label='ROC (area = {0:.4f})'.format(auc))
    print(fpr)


    print(tpr)
    # plt.title("class=0")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

