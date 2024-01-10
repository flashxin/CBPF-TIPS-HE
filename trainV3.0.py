import os.path
import cv2
import numpy as np
import yaml
import fun.makeLable as mklab
import matplotlib.pyplot as plt
import fun.dicom2npy as d2n
import fun.K_fold as kf
import model.UNet as U
import model.UandSPP
import fun.LoadDataset as Dload
import torch.utils.data as Data  # 批处理模块
import model.Resnet as md
from tqdm import tqdm
import wandb
import pickle
import torch
import torch.nn as nn
import math
from torchvision import transforms
import model.triDres as mc
import sys
import fun.lossfun as Loss

def weight_init(m):  # 初始化权重
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        # m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()

if __name__ == "__main__":
    with open('cfgV3.0.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        print('Read yaml Success')
        f.close()

    if cfg['GPUmode'] == 'custom':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['GPUseq']
    print('GPUmode is %s' % cfg['GPUmode'])
    # this config use in some Special circumstances
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:   %s' % device)
    lable, samplecount = mklab.GetHElable(cfg)
    trainsets, valsets = kf.K_fold(cfg, samplecount, lable)
    patients = list(lable.keys())
    if not cfg['npysave']:
        print('npy data have been creat!')
    else:
        d2n.dcm2npyV2(cfg,patients)
        print('.npy Save Success')

    EPOCH = cfg['epoch']
    BATCH_SIZE = cfg['batchsize']
    LR = cfg['lr']
    testFreq = cfg['testFreq']
    workers = cfg['workers']
    if cfg['wandbEnable'] == True:
        print('Use Wandb Save Log!!!')
        wandb.init(project=cfg['objname'], name=cfg['wandbname'])
    else:
        print('_______________________')
        print('WARRING: LOG without Save')
        print('_______________________')
    # net = md.doubleAttResModel(Achannel=int(cfg['slicenumber'][0]),Bchannel=int(cfg['slicenumber'][1]),kinds=cfg['kind'])

    for k in range(cfg['Kfold']):
        max_acc = cfg['baseacc']

        # net = md.doubleinAttResModel(Achannel=cfg['slicenumber'][0], Bchannel=cfg['slicenumber'][1], kinds=cfg['kind'])
        net = md.doubleAttResModel(Achannel=cfg['slicenumber'][0], Bchannel=cfg['slicenumber'][1], kinds=cfg['kind'])
        # net=md.ResModel(inchannel=cfg['slicenumber'][1],n_class=cfg['kind'])
        net.apply(weight_init)
        net.to(device)
        if (torch.cuda.device_count() > 1) and (cfg['GPUpara'] == True):
            net = torch.nn.DataParallel(net)
            print('Open GPUs Parallel Train')
        # optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
        optimizer=torch.optim.Adam(net.parameters(), lr=LR,weight_decay=0.0005)
        # loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
        loss_func=Loss.MyLoss(cfg)
        # loss_func=Loss.FocalLoss(class_num=2,alpha=torch.tensor([0.75,0.25]),size_average=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['decaystep'], gamma=cfg['lrdecay'])
        train_dataset = Dload.MyDataset(cfg, 'train', trainsets[k],lable,valsets[k])
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers,
                                       drop_last=True)
        val_dataset = Dload.MyDataset(cfg, 'val', valsets[k],lable,[])
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers,
                                     drop_last=False)
        for epoch in range(EPOCH):
            net.train()
            train_loader = tqdm(train_loader, file=sys.stdout)
            for step, (name, npysA,npysB) in enumerate(train_loader):
                # print('3')
                label = torch.zeros(len(name))
                for i in range(len(name)):
                    label[i] = lable[name[i]]
                # print(label)
                # print(label)
                # print(npy.shape)
                imgA = npysA.type(torch.FloatTensor)
                imgB = npysB.type(torch.FloatTensor)
                # img = train_data_transform(img)
                label = label.type(torch.LongTensor)
                imgA,imgB,label =imgA.to(device), imgB.to(device), label.to(device)
                output = net(imgA,imgB)
                # output = net(imgB)
                loss = loss_func(output, label,epoch)
                # loss.to(device)
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                scheduler.step()
                if step % 1 == 0:
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
                    if cfg['wandbEnable']:
                        wandb.log({'loss: %d-th fold' % k: loss,
                                   "Learn-rate %d-th fold" % k: optimizer.state_dict()['param_groups'][0]['lr']})
            if epoch % testFreq == 0:
                all_name = []
                all_label = []
                all_pred = []
                net.eval()
                with torch.no_grad():
                    val_loader = tqdm(val_loader, file=sys.stdout)
                    for step, (name, npysA,npysB) in enumerate(val_loader):
                        label = torch.zeros(len(name))
                        for i in range(len(name)):
                            # print(name[i])
                            label[i] = lable[name[i]]
                        imgA = npysA.type(torch.FloatTensor)
                        imgB = npysB.type(torch.FloatTensor)
                        # img = train_data_transform(img)
                        label = label.type(torch.LongTensor)
                        imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)
                        output = net(imgA,imgB)
                        # output = net(imgA)
                        pred = torch.max(output, 1)[1].data.cpu().numpy()

                        for i in range(len(name)):
                            # print(all_label)
                            all_name.append(name[i])
                            all_label.append(lable[name[i]])
                            all_pred.append(pred[i])
                    accuracy = float(
                        (all_pred == torch.tensor(all_label).data.cpu().numpy()).astype(int).sum()) / float(
                        torch.tensor(all_label).size(0))
                    print(all_pred)
                    print('| test acc %f.4  (%d -th fold)' % (accuracy, k))
                    if cfg['wandbEnable']:
                        wandb.log({'Epoch:%d' % k: epoch,
                                   'acc: %d-flod' % k: accuracy, })
                    if (np.mean(accuracy) > max_acc and epoch > 2):
                        max_acc = np.mean(accuracy)
                        torch.save(net.state_dict(), os.path.join(cfg['savepath'],  cfg['wandbname'] + '_%d_fold.pt' % k))
                        np.save(os.path.join(cfg['savepath'], cfg['wandbname'] + '_label_%d_fold.npy' % k),all_label)
                        np.save(os.path.join(cfg['savepath'],  cfg['wandbname'] + '_name_%d_fold.npy' % k), all_name)
                        np.save(os.path.join(cfg['savepath'], cfg['wandbname'] + '_pred_%d_fold.npy' % k), all_pred)
                        print('Save better weight success')