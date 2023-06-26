# -*- coding: utf-8 -*-
"""
本文件夹主要用来进行肿瘤抠出，并统一resize成一个大小
"""

import os
import cv2
from PIL import Image
from skimage import io
import scipy.ndimage
import numpy as np
import pandas as pd
import nrrd
import warnings
import csv
import time
from tqdm import tqdm
from tqdm._tqdm import trange


data_path = '/media/user/res/'
save_path = '/media/user/data0923/'
excel_path = save_path


data_crop = excel_path + 'pic_crop/'
data_npy_crop = excel_path + 'npy_crop/' #裁剪后要用的数据

np.set_printoptions(threshold=np.inf)




def mkdir(path):
    '''make dir'''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)



if not os.path.exists(data_npy_crop):
    os.makedirs(data_npy_crop + "image")
    os.makedirs(data_npy_crop + "mask")

if not os.path.exists(data_crop):
    os.makedirs(data_crop + "image")
    os.makedirs(data_crop + "mask")


##############################
img_name = os.listdir(data_path + 'ct/')
label_names = os.listdir(data_path + 'seg/')

a = pd.read_csv(data_path + 'log_280.csv', encoding="GB2312")
cln_data = np.array(a)

print(a)
print('================')
    

for ii in tqdm(range(len(label_names))):
    pat_name = "{}".format(str(img_name[ii].split('.nrrd')[-2]))
    print(pat_name)
    
    
    image, opt_i = nrrd.read(data_path + 'ct/' + img_name[ii])
    label, opt_l = nrrd.read(data_path + 'seg/' + label_names[ii])

    # print(type(image))   
    print(image.shape)
    # print(label.shape[2])

  
    #读取左上和右下两个位置的坐标
    result = a[a['Name']== pat_name].index.tolist()
    # print(len(result))
    
    for j in range(len(result)):
        # print(j)
        k = result[j]
        min_x = a.iloc[k,1]              
        min_y = a.iloc[k,2] 
        max_x = a.iloc[k,3] 
        max_y = a.iloc[k,4] 
        id_1  = a.iloc[k,5] - 1 

        # print(id_1)
        # print(max_y)
        # print(max_x)
        # print(min_y)
        # print(min_x)
        # print('================')

        img_slice = (image[:, :, id_1].T).astype(np.int16)
        label_slice = label[:, :, id_1].T   
        # print(img_slice.shape)


        img = img_slice[min_x:max_x, min_y:max_y]
        print(img.shape)
        print('++++++++++++++')
        mask = label_slice[min_x:max_x, min_y:max_y]


        img = cv2.resize(img,(224,224))
        print(img.shape)
        mask = cv2.resize(mask,(224,224))


        cv2.imwrite(data_crop + "image/{}_{}.png".format(pat_name, id_1 + 1),
                    (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.)
        cv2.imwrite(data_crop + "mask/{}_{}.png".format(pat_name, id_1 + 1), mask * 255.)

        np.save(data_npy_crop + "image/{}_{}.npy".format(pat_name, id_1 + 1), img)
        np.save(data_npy_crop + "mask/{}_{}_mask.npy".format(pat_name, id_1 + 1), mask * 255.)


