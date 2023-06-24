'''
Product Grad_Cam Heatmap
Paper https://arxiv.org/abs/1610.02391 
Copyright (c) Xiangzi Dai, 2020
python test_grad_cam.py
'''
import cv2
import numpy as np
import torch
from torch.autograd import Function
# from torchvision import models
import sys
from scipy.special import comb
from config_egfr_gui_1228 import opt
import os,fire
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torchnet import meter
from sklearn import metrics
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
from torch.autograd import Variable, Function
import models
import torch.nn.functional as F
import os


def get_last_conv(m):
    """
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    return list(convs)[-1]

class Grad_Cam:
    def __init__(self, model,target_layer_names, use_cuda):
        self.model = model
        self.target = target_layer_names
        self.use_cuda = use_cuda
        self.grad_val = []
        self.feature = [] #feature dim is same as grad_val
        self.hook = []
        self.img = []
        #self.img_2 = None#########
        self.inputs = None
        self._register_hook()
    def get_grad(self,module,input,output):
            self.grad_val.append(output[0].detach())
    def get_feature(self,module,input,output):
            self.feature.append(output.detach())
            #print(type(self.feature))
    def _register_hook(self):
        #for i in self.target:
        #print(self.get_feature.shape)
        i = self.model.bockbone
        self.hook.append(i.register_forward_hook(self.get_feature))
        self.hook.append(i.register_backward_hook(self.get_grad))

    def _normalize(self,cam):
        h,w,c = self.inputs.shape
        cam = (cam-np.min(cam))/np.max(cam)
        cam = cv2.resize(cam, (w,h))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap + np.float32(self.inputs)#######self.inputs)
        cam = cam / np.max(cam)
        return np.uint8(255*cam),np.uint8(255*heatmap)

    def remove_hook(self):
        for i in self.hook:
            i.remove()

    def _preprocess_image(self,img):
         means = [0.485, 0.456, 0.406]
         stds = [0.229, 0.224, 0.225]

         preprocessed_img = img.copy()[:, :, ::-1]
         for i in range(3):
             preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
             preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
         preprocessed_img = \
         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
         preprocessed_img = torch.from_numpy(preprocessed_img)
         preprocessed_img.unsqueeze_(0)
         input = preprocessed_img.requires_grad_(True)
         return input

    def __call__(self, img,idx=None):############
        """
        :param inputs: [w,h,c]
        :param idx: class id
        :return: grad_cam img list
        """
        self.model.zero_grad()
        
        self.inputs = np.float32(cv2.resize(img, (224, 224))) / 255
        inputs = self._preprocess_image(self.inputs)
        #self.img_2 = np.float32(cv2.resize(img2, (224, 224))) / 255#############
        #img2 = self._preprocess_image(self.img_2)##############
        if self.use_cuda:
            inputs = inputs.cuda()
            self.model = self.model.cuda()
        output, rec = self.model(inputs)
        #pro = t.nn.functional.softmax(output, dim=1)

        #cla_np = np.sum(pro[:,1].cuda().data.cpu().numpy()).reshape(-1,1)
        #lab_np = label.cuda().data.cpu().numpy()[0:1].reshape(-1,1)

        if idx is None:
            idx = np.argmax(output.detach().cpu().numpy()) #predict id
        target = output[0][idx]
        target.backward()
        #computer 
        weights = []
        for i in self.grad_val[::-1]: #i dim: [1,512,7,7]
             weights.append(np.mean(i.squeeze().cpu().numpy(),axis=(1,2)))
        for index,j in enumerate(self.feature):# j dim:[1,512,7,7]
             cam = (j.squeeze().cpu().numpy()*weights[index][:,np.newaxis,np.newaxis]).sum(axis=0)
             cam = np.maximum(cam,0) # relu
             cam,heatmap =self._normalize(cam)
             #self.img.append(self._normalize(cam))
             self.img.append(cam)
        return self.img


def data_cam(i,**kwargs):
    opt._parse(kwargs)
    
    # load model    
    model = getattr(models, opt.model)().eval()
    #model_path = "/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/0011-[1, 0.5]--0.0001--16/0094--1.0000--100.0000--0.7797--69.0476.pth"
    #model_path = "/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/012-[1, 0]--0.0001--16/0052--1.0000--97.6190--0.8315--78.5714.pth"
    model_path = '/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/011-[1, 0.6]--0.0001--16/0031--0.9844--92.5170--0.8267--76.1905.pth'
    #model_path ="/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/011-[1, 0]--0.0001--16/0022--0.9613--74.8299--0.8324--61.9048.pth"
    #model_path ="/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/0011-[1, 0.5]--0.0001--16/0038--0.9987--95.2381--0.8318--77.3810.pth"
    #model_path ="/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/012-[1, 0]--0.0001--16/0049--1.0000--99.6599--0.7872--69.0476.pth"
    model.load(model_path)
    model.to(opt.device)
    model = nn.DataParallel(model, device_ids=opt.device_ids)
    if isinstance(model,torch.nn.DataParallel):
		    model = model.module
    #for k in model.state_dict():
    #    print(k)
    #print(model)
    img_path = '/media/diskF/lar/code/song/img/pic/' + i # 训练集存放路径
    use_cuda = torch.cuda.is_available()
    #img2 = cv2.imread("/media/diskF/lar/code/song/img/pic/zhuxianlan.png",1)####

    img = cv2.imread(img_path, 1)
    #img3 = img[np.newaxis, ...]
    #data = np.concatenate((img3,img3,img3), axis=0)
    #data = t.FloatTensor(data)   
    i = i.split(".")[0]
    target_layer = ["bockbone.layer4.layers.15.dense_layer.5"] 
    Grad_cams = Grad_Cam(model,target_layer,use_cuda)
    grad_cam_list  = Grad_cams(img)###
    #print(grad_cam_list.shape)
    #target_layer corresponding grad_cam_list
    cv2.imwrite("/media/diskF/lar/code/song/img/cam0501/{}.png".format(i),grad_cam_list[0])
    #cv2.imwrite("/media/diskF/lar/code/song/img/{}.png".format(i),heatmap)
    cv2.imwrite("/media/diskF/lar/code/song/img/cam0314/{}.tiff".format(i),grad_cam_list[0],((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
                                                                                        int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
                                                                                        int(cv2.IMWRITE_TIFF_XDPI), 300,
                                                                                        int(cv2.IMWRITE_TIFF_YDPI), 300)))
	
	
if __name__ == '__main__':
    
    image_path = '/media/diskF/lar/code/song/img/pic/'
    img_list = os.listdir(image_path)
    for i in img_list:
      data_cam(i)











