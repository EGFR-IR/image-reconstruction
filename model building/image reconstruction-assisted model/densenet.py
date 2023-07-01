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
import BasicModule
import torch.nn.functional as F
import torch
import torchvision

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
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)

        return x

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
        self.rec = SConv(1024, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(   
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
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                DoubleConv(64, 64),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                DoubleConv(32, 32), 
                nn.Conv2d(32, 3, 1))

    def forward(self, x):
        x1 = self.bockbone(x)
        #print('x.shape',x.shape)  #[8, 3, 224, 224]
        #print('x1.shape',x1.shape) #[8, 1024, 7, 7]1664
        
        f1 = nn.functional.interpolate(x1, scale_factor=8, mode='bilinear', align_corners=True)
        #print('f1.shape',f1.shape) #[8, 1024, 56, 56]
        dense = self.avgpool(f1)
        #print('dense.shape',dense.shape)#[8, 1024, 1, 1]
        dense1 = dense.view(f1.size(0), 1024)    
        #print('dense1.shape',dense1.shape)  #[3, 1024]
        #print(dense1)       
        #fea = self.fc1(dense1)
        fea = self.fc(dense1)
        #print("fea shape:",fea.shape)
        cla_out = self.classifier(fea)
        #print("cla_out shape:",cla_out.shape)
        rec_fea = self.rec(f1)
        #print("rec_fea shape:",rec_fea.shape)  #[8, 128, 56, 56]                       
        re_1 = self.recons[0](rec_fea)
        #print("re_1 shape: ", re_1.shape)  #[64,112,112]      
        c11 = self.recons[1](re_1)
        #print("c11 shape: ", c11.shape)       #[64,112,112] 
        re_2 = self.recons[2](c11)
        #print("re_2 shape: ", re_2.shape)        #[32,224,224]
        c12 = self.recons[3](re_2)
        #print("c12 shape: ", c12.shape)        #[32,224,224]
        c13 = self.recons[4](c12)
        #print("c13 shape: ", c13.shape)        #[3,224,224]
        rec_out = nn.Sigmoid()(c13)
        #print("rec_out shape: ", rec_out.shape)

        return cla_out, rec_out





