from __future__ import print_function
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import cv2

import torch as t
from torch.utils import data
from torch.utils.data import DataLoader

import random

path = '/media/diskF/lar/data/'
path1 = '/media/diskF/lar/data/spilt_song/'
path2 = '/media/diskF/lar/data/'

random.seed(41)

class dataset_npy(data.Dataset):

    def __init__(self, root, root_mask, is_transform=None, train=False, val=False, test=False, exval=False):

        self.transforms = is_transform
        self.test = test
        self.train = train
        self.val = val
        self.exval = exval

        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        masks = [os.path.join(root_mask, mask) for mask in os.listdir(root_mask)]

        imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[-2]))
        #print(len(imgs))
        imgs = sorted(imgs, key=lambda x: x.split('/')[-1].split('_')[-2].split('-')[-1])
        masks = sorted(masks, key=lambda x: int(x.split('/')[-1].split('_mask')[-2].split('_')[-1]))
        masks = sorted(masks, key=lambda x: x.split('/')[-1].split('_mask')[-2].split('_')[-2].split('-')[-1])    
                    
        train_keys =   list(np.load(path1 + 'fold_1.npy')) \
                     + list(np.load(path1 + 'fold_2.npy')) \
                     + list(np.load(path1 + 'fold_3.npy')) \
                     + list(np.load(path1 + 'fold_4.npy')) \
                     + list(np.load(path1 + 'fold_5.npy')) \
                     + list(np.load(path1 + 'fold_6.npy')) \
                     + list(np.load(path1 + 'fold_7.npy')) 
        #print('train:',len(train_keys))
        # print(train_keys)
        # print("===============")
        test_keys =     list(np.load(path1 + 'fold_8.npy'))
        
        val_keys =      list(np.load(path1 + 'fold_9.npy'))\
                      + list(np.load(path1 + 'fold_10.npy'))
        
        exval_data = np.array(pd.read_csv(path + 'lab_87.csv', encoding="GB2312"))
        exval_name = exval_data[:, 0].tolist()
        exval_label = exval_data[:, 1].tolist()
        #print(exval_name,"lenname",len(exval_name))
        # print(val_keys)
        # print("=============================")
        clinical_data = np.array(pd.read_csv(path2 + 'lab_420_song.csv', encoding="GB2312"))
        pat_name = clinical_data[:, 0].tolist()
        label = clinical_data[:, 1].tolist()

        if self.exval:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in exval_name] 
            self.masks = [mask for mask in masks if mask.split('/')[-1].split('.')[-2].split('_')[-3] in exval_name]
            self.labels = [exval_label[exval_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in exval_name]
        elif self.test:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2].lower() in test_keys] 
            self.masks = [mask for mask in masks if mask.split('/')[-1].split('.')[-2].split('_')[-3].lower() in test_keys]
            self.labels = [label[pat_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2].lower())]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2].lower() in test_keys]
            #print(len(self.labels),len(self.imgs),len(self.masks))
        elif self.train:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2] in train_keys]
            self.masks = [mask for mask in masks if mask.split('/')[-1].split('.')[-2].split('_')[-3] in train_keys]
            self.labels = [label[pat_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2])]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2] in train_keys]
        elif self.val:
            self.imgs = [img for img in imgs if img.split('/')[-1].split('.')[-2].split('_')[-2].lower() in val_keys] 
            self.masks = [mask for mask in masks if mask.split('/')[-1].split('.')[-2].split('_')[-3].lower() in val_keys]
            self.labels = [label[pat_name.index(img.split('/')[-1].split('.')[-2].split('_')[-2].lower())]  for img in imgs
                           if img.split('/')[-1].split('.')[-2].split('_')[-2].lower() in val_keys]


    def __getitem__(self, index):
        img_path = self.imgs[index]
        mask_path = self.masks[index]
        label = self.labels[index]

        data = np.load(img_path)
        mask = np.load(mask_path)

        
        data = np.clip(data, -1000, 400)
        data = (data + 1000) / (1000 + 400)
        mask = mask / 255.
     

        data = data[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        
        # print(mask.shape)

        
        # data = np.expand_dims(data, axis=2)
        data = np.concatenate((data, data, data), axis=0)
        mask = np.concatenate((mask, mask, mask), axis=0)
        # print(len(data))
        # print('++++++')         
        data = t.FloatTensor(data)
        mask = t.FloatTensor(mask)

        # print(len(data))
        # print('===============')   


        if self.transforms:
     
            transform = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
            transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomChoice(transform),transforms.ToTensor()])
            data, mask  = transform(data), transform(mask)
            
           
        if self.transforms == 'feature':
            transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=1),transforms.ToTensor()])
            data = transform(data)
            mask = transform(mask)
        # print(type(data))
       
        return (data, mask), int(label)

    def __len__(self):
        return len(self.imgs)






