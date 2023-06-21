#本文件主要为vgg16_bn + 重建任务网络


# -*- coding:utf-8 -*-
import cv2
import time
import os
import torch.nn as nn
import matplotlib.pyplot as plt
# torchvision for pre-trained models
from torchvision import models
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.basic_module import BasicModule
import torch.nn.functional as F
import torch
import torchvision

'''
######################################################
savepath='/media/diskF/lar/code/song/heatmap/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))
'''
####################################################

__all__ = ['DenseNet121', 'DenseNet169','DenseNet201','DenseNet264']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class _TransitionLayer(BasicModule):
    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


class _DenseLayer(BasicModule):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(BasicModule):
    def __init__(self, num_layers, inplances, growth_rate, bn_size , drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseNet(BasicModule):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16],num_classes=2):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=3, places=init_channels)

        num_features = init_channels
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.layer1(x)
        #print('x1.shape',x1.shape)[3, 256, 56, 56]
        x2 = self.transition1(x1)
        x3 = self.layer2(x2)
        #print('x3.shape',x3.shape)[3, 512, 28, 28]
        x4 = self.transition2(x3)
        x5 = self.layer3(x4)
        #print('x5.shape',x5.shape)([3, 1024, 14, 14])
        x6 = self.transition3(x5)
        x7 = self.layer4(x6)
        #print('x7.shape',x7.shape)[3, 1024, 7, 7]

        return x1,x3,x5,x7

def DenseNet121():
    model = DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16])
    # return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16])
    #print(model)
    return model

def DenseNet169():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 32, 32])

def DenseNet201():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 48, 32])

def DenseNet264():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 64, 48])
##############################################


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
        
#重建：
class SConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, 1, padding=1),
            nn.Conv2d(in_ch, out_ch, 1),
            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class MDenseNet(BasicModule):
    def __init__(self):
        super(MDenseNet, self).__init__()
        self.bockbone = DenseNet121()
        self.rec = SConv(2816, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(    # 定义自己的分类层  
                #nn.Linear(1024, 1024),
                #nn.ReLU(True),
                #nn.Dropout(0.4)  ,            
                nn.Linear(1024, 512)  ,
                nn.Sigmoid(),
                nn.Dropout(0.4),
                # nn.Linear(512, num_classes),
                )
        self.classifier = nn.Linear(512, 2)
        self.recons = nn.Sequential(  
                #nn.ConvTranspose2d(512, 256, 2, stride=2),
                #DoubleConv(256, 256),
                #nn.ConvTranspose2d(256, 128, 2, stride=2),
                #DoubleConv(128, 128),             
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                DoubleConv(64, 64),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                DoubleConv(32, 32), 
                nn.Conv2d(32, 3, 1))

    def forward(self, x):
        x1,x3,x5,x7 = self.bockbone(x)
        #print('x.shape',x.shape)  #[8, 3, 224, 224]
        # print('x1.shape',x1.shape) #[8, 1024, 7, 7]
        
        f1 = nn.functional.interpolate(x7, scale_factor=8, mode='bilinear', align_corners=True)#[1024, 56, 56]
        #f2 = nn.functional.interpolate(x5, scale_factor=4, mode='bilinear', align_corners=True)#1024, 56,56
        #f3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)#512, 56,56
        #f4 = nn.functional.interpolate(x1, scale_factor=1, mode='bilinear', align_corners=True)#256, 56, 56
        #f_all = torch.cat([f1,f2,f3,f4],dim=1) 
        #print('f_all.shape',f_all.shape) #[8, 1024, 56, 56][8, 2816, 56, 56]
        dense = self.avgpool(f1)
        #print('dense.shape',dense.shape)#[8, 1024, 1, 1]
        dense1 = dense.view(f1.size(0), 1024)    
        #print('dense1.shape',dense1.shape)  #[3, 1024]
        #print(dense1)       
        #fea = self.fc1(dense1)
        fea = self.fc(dense1)
        #print("fea shape:",fea.shape)
        cla_out = self.classifier(fea)# 自定义的分类部分 
        #print("cla_out shape:",cla_out.shape)
        
        ###################################################重建#################
        rec_fea = self.rec(f_all)
        #print("rec_fea shape:",rec_fea.shape)  #[8, 128, 56, 56]                       
        re_1 = self.recons[0](rec_fea)
        #print("re_1 shape: ", re_1.shape)  #[64,112,112]      
        c11 = self.recons[1](re_1)
        #print("c11 shape: ", c11.shape)       #[64,112,112] 
        re_2 = self.recons[2](c11)
        #print("re_2 shape: ", re_2.shape)        #[32,224,224]
        c12 = self.recons[3](re_2)
        #print("c12 shape: ", c12.shape)        #[32,224,224]
        #re_3 = self.recons[4](c12)
        #c13 = self.recons[5](re_3)
        #re_4 = self.recons[6](c13)
        #c14 = self.recons[7](re_4)
        c13 = self.recons[4](c12)
        #print("c15 shape: ", c15.shape)        #[3,224,224]
        rec_out = nn.Sigmoid()(c13)
        #print("rec_out shape: ", rec_out.shape)
        #classes_id = np.argmax(cla_out.cpu().numpy())
        #cam = (dense1.cpu().numpy())[0]
        #cam = abs(classes_id-cam)
        #cam = cam.dot(((x1.cpu().numpy())[0]).reshape(1024, 7*7))
        #cam = cam.reshape(7, 7)
        #cam = (cam - cam.min()) / (cam.max() - cam.min())
        #cam_gray = np.uint8(255 * cam)
        
        #print("cam shape",cam.shape)
        return cla_out, rec_out
        #return cla_out, rec_out, cv2.resize(cam_gray, (224, 224))





