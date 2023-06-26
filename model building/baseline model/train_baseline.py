#此文件主要为使用resnet50作为编码器，构建解码器，实现分类+重建辅助任务


# -*- coding: utf-8 -*
'''
  CUDA_VISIBLE_DEVICES=0,1  python3 train.py train 1 0.5  --use-gpu --ex-num='001-[1, 0.5]'
'''
from scipy.special import comb
import sys
from config_egfr import opt
import os,fire
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torchnet import meter
from dataset_egfr import dataset_npy
from utils.dice_loss import DiceCoeff, dice_coeff
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
from sklearn import metrics
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import pandas as pd
from torch.autograd import Variable, Function
import densenet
import torch.nn.functional as F
import cv2

###################################
#################################################################################
#标签平滑
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
        
        
# 三个loss:分类，重建，一致性
#cla_criterion = nn.CrossEntropyLoss() #分类交叉熵loss
cla_criterion = LabelSmoothing()
rec_criterion = nn.MSELoss()          #重建MSE loss
con_criterion = nn.KLDivLoss()        #一致性KL散度loss

a = int(sys.argv[2])
c = int(float(sys.argv[3]))
#print(a,c)
name = 'res50+ssl_20230201'

def train(**kwargs):
    opt._parse(kwargs)
    #vis = Visualizer(opt.env, port=opt.vis_port)    #可视化

    
    if not os.path.exists('./result20230201/bt/checkpoints_' + name +'/{}_multi/{}--{}--{}/'.format(opt.model, opt.ex_num, opt.lr, opt.batch_size)):
        os.makedirs('./result20230201/bt/checkpoints_' + name +'/{}_multi/{}--{}--{}/'.format(opt.model, opt.ex_num, opt.lr, opt.batch_size))
    
    # step1: configure model
    model = getattr(densenet, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)    
    model.to(opt.device)
    model = nn.DataParallel(model, device_ids=opt.device_ids)
    
    
    # step2: data
    train_data = dataset_npy(opt.train_data_root,opt.train_data_label,is_transform = True, train=True)
    test_data = dataset_npy(opt.test_data_root,opt.test_data_label,is_transform = False, test=True)
    val_data = dataset_npy(opt.val_data_root,opt.val_data_label,is_transform = False, val=True)
    exval_data = dataset_npy(opt.exval_data_root,opt.exval_data_label,is_transform = False, exval=True)
    
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    exval_dataloader = DataLoader(exval_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)


    # step3: criterion and optimizer
    lr = opt.lr

    # step4: meters
    loss_meter_1 = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    #构建结果文件
    submit_file_name = 'results.csv'
    submit_path = './result20230201/bt/checkpoints_' + name +'/{}_multi/{}--{}--{}/'.format(opt.model, opt.ex_num, opt.lr, opt.batch_size)
    csv_file = open(submit_path + submit_file_name, 'w')
    csv_file.write("epoch, te_auc, te_acc, te_sens, te_spec, va_auc, va_acc, va_sens, va_spec, exva_auc, exva_acc, exva_sens, exva_spec,\n")


    # train    
    for epoch in range(opt.max_epoch):
        model.train()     
 
        loss_meter_1.reset()
        confusion_matrix.reset()
        
        for ii, ((data, mask),label) in tqdm(enumerate(train_dataloader)):
            data = data.to(opt.device)

            label = t.LongTensor(label).to(opt.device)
                       
            cla_out = model(data)
            
            loss_1 = cla_criterion(cla_out, label)

            # 多loss整合
            loss = a * loss_1

          
            params = filter(lambda p: p.requires_grad, model.parameters())                      
            optimizer = t.optim.Adam(params, lr, betas=(0.9, 0.99))     
            
            optimizer.zero_grad()           
            loss.backward()
            optimizer.step()

            loss_meter_1.add(loss_1.item())


            if (ii + 1) % opt.print_freq == 0:
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        # validate and visualize
        loss_cla_train = loss_meter_1.value()[0]

          
        pat_train_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, train=True)
        pat_train_dataloader = DataLoader(pat_train_data, 3, shuffle=False, num_workers=opt.num_workers)
        pat_test_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, test=True)
        pat_test_dataloader = DataLoader(pat_test_data, 3, shuffle=False, num_workers=opt.num_workers)
        pat_val_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, val=True)
        pat_val_dataloader = DataLoader(pat_val_data, 3, shuffle=False, num_workers=opt.num_workers)        
        pat_exval_data = dataset_npy(opt.exval_data_root, opt.exval_data_label, is_transform=False, exval=True)
        pat_exval_dataloader = DataLoader(pat_exval_data, 3, shuffle=False, num_workers=opt.num_workers)


        tr_auc, tr_acc, tr_sens, tr_spec, tr_cla, tr_rec, tr_lab = val(model, pat_train_dataloader)
        te_auc, te_acc, te_sens, te_spec, te_cla, te_rec, te_lab = val(model, pat_test_dataloader)
        va_auc, va_acc, va_sens, va_spec, va_cla, va_rec, va_lab = val(model, pat_val_dataloader)
        exva_auc, exva_acc, exva_sens, exva_spec, exva_cla, exva_score, exva_lab = val(model, pat_exval_dataloader)
   
        model.module.save(
            './result20230201/bt/checkpoints_' + name + '/{}_multi/{}--{}--{}/{:0>4d}--{:.4f}--{:.4f}--{:.4f}--{:.4f}.pth'.format(opt.model,
                                                                                                       opt.ex_num,
                                                                                                       opt.lr,
                                                                                                       opt.batch_size,
                                                                                                       epoch, tr_auc,
                                                                                                       tr_acc, va_auc,
                                                                                                       va_acc))

        csv_file.write(str(epoch) + ',' + str(te_auc) + ',' + str(te_acc) + ',' + str(te_sens) + ',' + str(te_spec) +
                                    ',' + str(va_auc) + ',' + str(va_acc) + ',' + str(va_sens) + ',' + str(va_spec) +
                                    ',' + str(exva_auc) + ',' + str(exva_acc) + ',' + str(exva_sens) + ',' + str(exva_spec) +  '\n')
                                    
                ##保存预测打分和金标准（测试集和验证集）
        np.savetxt('./result20230201/bt/checkpoints_' + name + '/{}_multi/{}--{}--{}/{:0>4d}_val.csv'.format(opt.model, opt.ex_num, opt.lr,
                                                                           opt.batch_size, epoch),
                   np.concatenate([va_score, va_lab], axis=1), delimiter=',')

        np.savetxt(
            '././result20230201/bt/checkpoints_' + name + '/{}_multi/{}--{}--{}/{:0>4d}_exval.csv'.format(opt.model, opt.ex_num, opt.lr,
                                                                     opt.batch_size, epoch),
            np.concatenate([exva_score, exva_lab], axis=1), delimiter=',')

   
        # update learning rate
        if epoch % 20 == 0 and epoch != 0:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print('******************Epoch: ',epoch,'******************')
        print('train_cla_loss:',loss_cla_train)
        print('val_cla_loss:',va_cla) 
        print('val_rec_loss:',va_rec)
        print('train_auc:{};test_auc:{};val_auc:{};exval_auc:{}'.format(tr_auc,te_auc,va_auc,exva_auc))      
        print('train_acc:{};test_auc:{};val_auc:{};exval_auc:{}'.format(tr_acc,te_acc,va_acc,exva_acc))


@t.no_grad()
def val(model, dataloader, heatmap=False):
    """
    计算模型在验证集上的准确率等信息
    return: 分割损失，重分割损失，重建损失，一致性损失
    """
    if isinstance(model,t.nn.DataParallel):
        model = model.module
    model.eval()


    cla = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    # 分类的打分和金标准
    score = np.array([]).reshape(0,1)
    lab = np.array([]).reshape(0,1)
    count = 0
    # score = []
    # lab   = []
    
    for ii, ((val_input, mask),label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        mask = mask.to(opt.device)
        label = label.to(opt.device)
          
        #把预测为第一类的概率和对应标签转换成数组
        cla_score = model(val_input)

        pro = t.nn.functional.softmax(cla_score, dim=1)

        cla_np = np.sum(pro[:,1].cuda().data.cpu().numpy()).reshape(-1,1)
        lab_np = label.cuda().data.cpu().numpy()[0:1].reshape(-1,1)

        score = np.concatenate([score,cla_np],0)
        lab = np.concatenate([lab,lab_np],0)

        #每个病人取平均计算混淆矩阵
        confusion_matrix.add(t.mean(pro,dim=0).view(1,2).detach(), label[0:1].type(t.LongTensor))

        cla_loss  = cla_criterion(cla_score, label)

        cla.add(cla_loss.item())


        
    model.train()
 
    #cla
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    sens_c = cm_value[0][0]/(cm_value[0][0] + cm_value[0][1])
    spec_c = cm_value[1][1]/(cm_value[1][1] + cm_value[1][0])    
    
    AUC = roc_auc_score(lab, score)
    return AUC, accuracy, sens_c, spec_c, cla.value()[0],score,lab


def val_print(**kwargs):
    opt._parse(kwargs)
    model = getattr(densenet, opt.model)().eval()

    #模型所在目录
    model_path = "/model.pth"

    model.load(model_path)
    model.to(opt.device)
    model = nn.DataParallel(densenet, device_ids=opt.device_ids)
    
    pat_train_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, test=True)
    pat_train_dataloader = DataLoader(pat_train_data, 3, shuffle=False, num_workers=opt.num_workers)

    pat_val_data = dataset_npy(opt.exval_data_root, opt.exval_data_label, is_transform=False, exval=True)
    pat_val_dataloader = DataLoader(pat_val_data, 3, shuffle=False, num_workers=opt.num_workers)

    va_auc, va_acc, va_sens, va_spec, va_cla, va_score, va_lab = val(model, pat_val_dataloader)

    print(str(va_auc) + ',' + str(va_acc) + ',' + str(va_sens) + ',' + str(va_spec) + ',' + '\n')


def help():
    """
    打印帮助的信息： python file.py help
    """
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__ == '__main__':
    fire.Fire()




