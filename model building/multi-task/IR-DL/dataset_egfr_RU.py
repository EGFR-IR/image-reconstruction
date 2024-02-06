# -*- coding: utf-8 -*
#CUDA_VISIBLE_DEVICES=0,1 python3 dataset_nsclc_gui.py  --use-gpu

from __future__ import print_function
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import cv2
from sklearn import preprocessing
import torch as t
from torch.utils import data
from torch.utils.data import DataLoader

import random
#/media/user/Disk01/liaoran/code/data_code_egfr_0316/data_code_egfr_0316/data/npy_crop_280_1228/
#/media/user/Disk01/liaoran/code/data_code_egfr_0316/data_code_egfr_0316/data/spilt_0923/
#/media/user/Disk01/liaoran/code/data_code_egfr_0316/data_code_egfr_0316/data/
# path = '/media/user/Disk01/guidongqi/egfr_vgg16_0923/data0923/npy_crop_224/'
#/media/diskF/lar/data/spilt_450/
#/media/diskF/lar/data/data_775_405/
path = 'res_unet/data_663_355_162/'

path1 = 'res_unet/spilt_song/'

path2 = 'res_unet/'

random.seed(41)

class dataset_npy(data.Dataset):

    def __init__(self, root, is_transform=None, train=False, val=False, test=False, exval=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试，以及外部测试集划分数据
        """
        self.transforms = is_transform
        self.test = test
        self.train =  train
        self.val = val
        self.exval = exval

        imgs = [os.path.join(root, img) for img in os.listdir(root)]   #
        #masks = [os.path.join(root_mask, mask) for mask in os.listdir(root_mask)]

        # imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-1]))
        # imgs = sorted(imgs, key=lambda x: int(x.split('_')[-2].split('-')[-1]))
        # masks = sorted(masks, key=lambda x: int(x.split('_mask')[-2].split('_')[-1]))
        # masks = sorted(masks, key=lambda x: int(x.split('_mask')[-2].split('_')[-2].split('-')[-1]))

        imgs = sorted(imgs, key=lambda x: (x.split('/')[-1].split('.')[-2].split('_')[-2]))
        #print(len(imgs))
        #imgs = sorted(imgs, key=lambda x: x.split('/')[-1].split('_')[-2].split('-')[-1])
        #masks = sorted(masks, key=lambda x: int(x.split('/')[-1].split('_mask')[-2].split('_')[-1]))
        #masks = sorted(masks, key=lambda x: x.split('/')[-1].split('_mask')[-2].split('_')[-2].split('-')[-1])    
        # print(len(imgs))# 按照7:1:2比例划分训练，测试，验证集
                    
        train_data =  np.array(pd.read_csv(path + 'lab_433.csv', encoding="GB2312"))
        train_name = train_data[:, 0].tolist()
        train_label = train_data[:, 1].tolist()
        #print('train:',len(train_keys))
        # print(train_keys)
        # print("===============")
        test_data =  np.array(pd.read_csv(path + 'lab_186.csv', encoding="GB2312"))          
        test_name = test_data[:, 0].tolist()
        test_label = test_data[:, 1].tolist()
        
        val_data =  np.array(pd.read_csv(path + 'lab_292.csv', encoding="GB2312"))    
        val_name = val_data[:, 0].tolist()
        val_label = val_data[:, 1].tolist()
        
        exval_data = np.array(pd.read_csv(path + 'lab_152.csv', encoding="GB2312"))
        exval_name = exval_data[:, 0].tolist()
        exval_label = exval_data[:, 1].tolist()
        #print(exval_name,"lenname",len(exval_name))
        # print(val_keys)
        # print("=============================")
        if self.train:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in train_name] 
            self.labels = [train_label[train_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in train_name]
        if self.test:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in test_name] 
            self.labels = [test_label[test_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in test_name]                    
        if self.val:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in val_name] 
            self.labels = [val_label[val_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in val_name]
        if self.exval:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in exval_name] 
            self.labels = [exval_label[exval_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in exval_name]

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        #mask_path = self.masks[index]
        label = self.labels[index]

        data = np.load(img_path)
        #mask = np.load(mask_path)
        # print(len(data))
        
        # print("data:",data.shape)
        # print('===============') 
        # data = cv2.resize(data,(224,224))
        # mask = cv2.resize(mask,(224,224))
        # print("mask:",mask.shape) 

        # print(data.shape) 

        #
        data = np.clip(data, -1000, 400)
        data = (data + 1000) / (1000 + 400)
        #mask = mask / 255.
        #data = data / 255.
        # z-score
        #data = preprocessing.scale(data)      
        #print("1")
        data = data[np.newaxis, ...]
        #mask = mask[np.newaxis, ...]
        
        # print(mask.shape)

        #转换为3通道
        
        # data = np.expand_dims(data, axis=2)
        data = np.concatenate((data, data, data), axis=0)
        #mask = np.concatenate((mask, mask, mask), axis=0)
        # print(len(data))
        # print('++++++')   


              
        data = t.FloatTensor(data)
        
        #mask = t.FloatTensor(mask)

        # print(len(data))
        # print('===============')   


        if self.transforms:
     
            transform = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
            transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomChoice(transform),transforms.ToTensor()])
            #data, mask  = transform(data), transform(mask)
            data  = transform(data)
            
           
        if self.transforms == 'feature':
            transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=1),transforms.ToTensor()])
            data = transform(data)
            #mask = transform(mask)
        # print(type(data))
       
        return (data), int(label)

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train_data_root = '/media/diskF/lar/data/data_663_355_162/663/'  # 训练集存放路径
    test_data_root = '/media/diskF/lar/data/data_663_355_162/663/'   # 测试集存放路径
    val_data_root = '/media/diskF/lar/data/data_663_355_162/355/'  # 外部验证集1存放路径
    exval_data_root = '/media/diskF/lar/data/data_663_355_162/ACM/'  # 外部验证集2存放路径
    
    
    print("************************************************")

    train_data = dataset_npy(train_data_root, train_data_mask,is_transform = True, val=True)
    # print(len(train_data))
    train_dataloader = DataLoader(train_data, batch_size = 1, shuffle=True, num_workers=2)


    # for ii, ((val_input, mask),label) in enumerate(train_dataloader):
        # print(len(data))
        # print(mask.shape)
       # print(data.shape)





