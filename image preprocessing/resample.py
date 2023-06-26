##CUDA_VISIBLE_DEVICES=3 python3 resample_chest.py  --use-gpu
#通过三线性插值将体积数据调整为1×1×1 mm的间距

import os
import cv2
from PIL import Image
from skimage import io
import scipy.ndimage
import numpy as np
import pandas as pd
import nrrd
import warnings
import time
from tqdm import tqdm
from tqdm._tqdm import trange


def resample(imgs_to_process,  patient, spacing, new_spacing, order):
    
    image = imgs_to_process
    scan = patient
    # Determine current pixel spacing
    spacing = spacing
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor   
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=order)
    
    return image, new_spacing

warnings.filterwarnings("ignore")



data_path = '/media/user/ct/'#nrrd数据位置
excel_path = '/media/user/data_pro/'
# print(os.listdir(excel_path))
clinical = np.array(pd.read_csv(excel_path + 'data_info.csv', encoding='gb18030'))

pat_name = clinical[:, 0]
img_name = pat_name + '.nrrd'


data_new = excel_path + 'data_res/'
if not os.path.exists(data_new):
        os.makedirs(data_new + "ct")


for i in tqdm(range(len(pat_name))):
    spacing = [clinical[i, 2], clinical[i, 3], clinical[i, 1]]
    # print(spacing)
    name_temp = img_name[i].split('.nrrd')[-2]
    
    print('name:', name_temp)
    img_temp = name_temp + '.nrrd'
    img, opt_i = nrrd.read(data_path + img_temp)
    print("HU值：", img)
    print("HU值2：", type(img))
        
    
    imgs_to_process = np.clip(img, -1024, 3071)
    print("处理后的Hu值：", imgs_to_process.shape)
    
    imgs_after_resamp, spacing_img = resample(imgs_to_process, img, spacing, [1,1,1], 3)  
    

    print("Shape before resampling\t", imgs_to_process.shape) #Shape before resampling  (129, 512, 512)
    print("Shape after resampling\t", imgs_after_resamp.shape) #Shape after resampling   (206, 292, 29)
    

    nrrd.write(data_new + "ct" + "/{}".format(img_temp), imgs_after_resamp)


