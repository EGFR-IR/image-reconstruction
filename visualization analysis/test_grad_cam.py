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
import sys
from scipy.special import comb
from config_egfr import opt
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
import densenet
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
        if self.use_cuda:
            inputs = inputs.cuda()
            self.model = self.model.cuda()
        output, rec = self.model(inputs))

        if idx is None:
            idx = np.argmax(output.detach().cpu().numpy()) #predict id
        target = output[0][idx]
        target.backward()

        weights = []
        for i in self.grad_val[::-1]:
             weights.append(np.mean(i.squeeze().cpu().numpy(),axis=(1,2)))
        for index,j in enumerate(self.feature):#
             cam = (j.squeeze().cpu().numpy()*weights[index][:,np.newaxis,np.newaxis]).sum(axis=0)
             cam = np.maximum(cam,0)
             cam,heatmap =self._normalize(cam)
             self.img.append(cam)
        return self.img


def data_cam(i,**kwargs):
    opt._parse(kwargs)
    
    # load model    
    model = getattr(densenet, opt.model)().eval()
    model_path ="/media/diskF/lar/code/song/model.pth"
    model.load(model_path)
    model.to(opt.device)
    model = nn.DataParallel(model, device_ids=opt.device_ids)
    if isinstance(model,torch.nn.DataParallel):
		    model = model.module
    img_path = '/media/diskF/lar/code/song/img/pic/' + i 
    use_cuda = torch.cuda.is_available()

    img = cv2.imread(img_path, 1)
    i = i.split(".")[0]
    target_layer = ["bockbone.layer4.layers.15.dense_layer.5"] 
    Grad_cams = Grad_Cam(model,target_layer,use_cuda)
    grad_cam_list  = Grad_cams(img)###
    cv2.imwrite("/media/diskF/lar/code/song/img/cam0501/{}.png".format(i),grad_cam_list[0])
	
if __name__ == '__main__':
    #
    image_path = '/media/diskF/lar/code/song/img/pic/'
    img_list = os.listdir(image_path)
    for i in img_list:
      data_cam(i)











