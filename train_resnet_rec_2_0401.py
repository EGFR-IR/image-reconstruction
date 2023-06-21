#此文件主要为使用resnet50作为编码器，构建解码器，实现分类+重建辅助任务


# -*- coding: utf-8 -*
'''
  CUDA_VISIBLE_DEVICES=0,1  python3 train_resnet_rec_2_1228.py train 1 0.5  --use-gpu --ex-num='001-[1, 0.5]'#23451 42513
  CUDA_VISIBLE_DEVICES=0,1  python3 train_resnet_rec_2_0401.py val_print 1 0  --use-gpu --ex-num='001-[1, 0]'#23451 42513


'''
from scipy.special import comb
import sys
from config_egfr_gui_1228 import opt
import os,fire
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torchnet import meter
from dataset.dataset_egfr_gui_1228 import dataset_npy
from utils.dice_loss import DiceCoeff, dice_coeff
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
from sklearn import metrics
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import pandas as pd
from torch.autograd import Variable, Function
import models
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

#[a, c, e] = [1, 0.6, 0]       #三个loss对应的加权系数
#print(sys.argv[2],type(sys.argv[2]))
#[a, c] = int(sys.argv[2])       #2个loss对应的加权系数
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
    model = getattr(models, opt.model)()
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
    # [loss_meter_1] = [meter.AverageValueMeter() for i in range(1)]
    #[loss_meter_1, loss_meter_3, loss_meter_5] = [meter.AverageValueMeter() for i in range(3)]
    [loss_meter_1, loss_meter_3] = [meter.AverageValueMeter() for i in range(2)]
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
        loss_meter_3.reset()
        #loss_meter_5.reset()
        confusion_matrix.reset()
        
        for ii, ((data, mask),label) in tqdm(enumerate(train_dataloader)):
            # train model:
            #data = t.stack(data, dim=1)
            data = data.to(opt.device)
            
            
            # print(data.shape)
            # print('++++++++++')
            # mask = mask.to(opt.device)
            label = t.LongTensor(label).to(opt.device)
                       
            cla_out,    rec_out = model(data)
            #print('rec_out.shape:',rec_out.shape,'data.shape',data.shape)
            #cla_out_2, re_out_2, fea_2 = model(rec_out)
            
            loss_1 = cla_criterion(cla_out, label)
            loss_3 = rec_criterion(rec_out, data)
            #loss_5 = con_criterion(fea, fea_2)
            # loss_5 = con_criterion(fea_2, fea)

            # # 多loss整合
            # loss = a*loss_1 + c*loss_3 + e*loss_5
            # # loss = a*loss_1 + c*loss_3  

            # 多loss整合
            loss = a * loss_1 + c * loss_3

          
            params = filter(lambda p: p.requires_grad, model.parameters())                      
            optimizer = t.optim.Adam(params, lr, betas=(0.9, 0.99))     
            
            optimizer.zero_grad()           
            loss.backward()
            optimizer.step()

            loss_meter_1.add(loss_1.item())
            loss_meter_3.add(loss_3.item())
            #loss_meter_5.add(loss_5.item())


            if (ii + 1) % opt.print_freq == 0:
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        # validate and visualize
        loss_cla_train = loss_meter_1.value()[0]
        loss_rec_train = loss_meter_3.value()[0]
        #loss_con_fea = loss_meter_5.value()[0]
       
          
        pat_train_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, train=True)
        pat_train_dataloader = DataLoader(pat_train_data, 3, shuffle=False, num_workers=opt.num_workers)
        pat_test_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, test=True)
        pat_test_dataloader = DataLoader(pat_test_data, 3, shuffle=False, num_workers=opt.num_workers)
        pat_val_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, val=True)
        pat_val_dataloader = DataLoader(pat_val_data, 3, shuffle=False, num_workers=opt.num_workers)        
        pat_exval_data = dataset_npy(opt.exval_data_root, opt.exval_data_label, is_transform=False, exval=True)
        pat_exval_dataloader = DataLoader(pat_exval_data, 1, shuffle=False, num_workers=opt.num_workers)

        # tr_auc, tr_acc, tr_sens, tr_spec, tr_dice_mean, tr_dice_std, tr_dice_min, tr_dice_max, \
        # tr_ppv_mean, tr_ppv_std, tr_sens_mean, tr_sens_std = val(model, pat_train_dataloader)

        # va_auc, va_acc, va_sens, va_spec, va_dice_mean, va_dice_std, va_dice_min, va_dice_max, \
        # va_ppv_mean, va_ppv_std, va_sens_mean, va_sens_std = val(model, pat_val_dataloader)


        tr_auc, tr_acc, tr_sens, tr_spec, tr_cla, tr_rec, tr_score, tr_lab = val(model, pat_train_dataloader)
        te_auc, te_acc, te_sens, te_spec, te_cla, te_rec, te_score, te_lab = val(model, pat_test_dataloader)
        va_auc, va_acc, va_sens, va_spec, va_cla, va_rec, va_score, va_lab = val(model, pat_val_dataloader)
        exva_auc, exva_acc, exva_sens, exva_spec, exva_cla, exva_rec, exva_score, exva_lab = val(model, pat_exval_dataloader)
        exva_auc, exva_acc, exva_sens, exva_spec, exva_score, exva_lab = exval(exva_score)
   
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
            np.array(list(zip(exva_score, exva_lab))), delimiter=',')
                   
        #vis.plot('train_cla_loss',loss_cla_train) 
        # vis.plot('train_rec_loss',loss_rec_train) 
        # vis.plot('train_con_fea_loss:',loss_con_fea) 
        #vis.plot('val_cla_loss',va_cla) 
        # vis.plot('val_rec_loss',va_rec) 
        # vis.plot('val_con_fea_loss:',va_con_fea) 
                                   
        #vis.plot('train_cla_auc', tr_auc)
        #vis.plot('train_cla_acc', tr_acc)               
        #vis.plot('val_cla_auc',va_auc) 
        #vis.plot('val_cla_acc', va_acc)

        # vis.log("epoch:{epoch},lr:{lr},train_cla:{train_cla},train_rec:{train_rec},tr_cla1:{tr_cla1},tr_rec:{tr_rec},\
                # va_cla1:{va_cla1},va_rec:{va_rec},tr_acc:{tr_acc},va_acc:{va_acc},tr_auc:{tr_auc},va_auc:{va_auc}".format(epoch=epoch, lr=lr, \
                # train_cla=loss_cla_train, train_rec=loss_rec_train, tr_cla1=tr_cla, tr_rec=tr_rec,va_cla1 = va_cla, va_rec=va_rec,
                # tr_acc = tr_acc,va_acc = va_acc,tr_auc = tr_auc,va_auc = va_auc))
                   
   
        # update learning rate
        if epoch % 20 == 0 and epoch != 0:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print('******************Epoch: ',epoch,'******************')
        print('train_cla_loss:',loss_cla_train)       
        print('train_rec_loss:',loss_rec_train)
        #print('train_con_fea_loss:',loss_con_fea)
        print('val_cla_loss:',va_cla) 
        print('val_rec_loss:',va_rec)
        #print('val_con_fea_loss:',va_con_fea)
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
    rec = meter.AverageValueMeter()
    con_fea = meter.AverageValueMeter()
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
        # cla_score = model(val_input)        
        #cla_score,  rec_score, x1 = model(val_input)
        cla_score,  rec_score = model(val_input)
        #x1 = cv2.applyColorMap(cv2.resize(x1, (224,224)),cv2.COLORMAP_JET)
        img = (val_input.cpu().numpy())[0]
        img_rec = (rec_score.cpu().numpy())[0]
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        img = np.uint8(255 * img)
        img_rec = img_rec.swapaxes(0, 1)
        img_rec = img_rec.swapaxes(1, 2)
        img_rec = np.uint8(255 * img_rec)
        #img_rec = img * 0.5 + img_rec * 0.5
        #fusionimg = img * 0.6 + x1 * 0.4
        #print(fusionimg)
        #cla_score2, rec_score2 , fea2 = model(rec_score)
        # fea = F.log_softmax(fea,dim=-1)
        # fea2 = fea2.log()
        # fea2 =F.softmax(fea2,dim=-1)       
        # print("fea_r shape:", fea)
        # print("fea_2_r shape:", fea2)
        # fea = F.log_softmax(fea,dim=-1)
        # fea2 = fea2.log()
        # fea2 =F.softmax(fea2,dim=-1)       
        # print("fea_r shape:", fea)
        # print("fea_2_r shape:", fea2)
        
        # pro = t.nn.functional.softmax\
            # (cla_score)[:,1].data.tolist()               

        pro = t.nn.functional.softmax(cla_score, dim=1)

        cla_np = np.sum(pro[:,1].cuda().data.cpu().numpy()).reshape(-1,1)
        lab_np = label.cuda().data.cpu().numpy()[0:1].reshape(-1,1)

        score = np.concatenate([score,cla_np],0)
        lab = np.concatenate([lab,lab_np],0)

        #每个病人取平均计算混淆矩阵
        confusion_matrix.add(t.mean(pro,dim=0).view(1,2).detach(), label[0:1].type(t.LongTensor))

        cla_loss  = cla_criterion(cla_score, label)
        rec_loss = rec_criterion(rec_score,val_input)
        #con_fea_loss = con_criterion(fea2,fea)

        cla.add(cla_loss.item())
        rec.add(rec_loss.item())
        #con_fea.add(con_fea_loss.item())
        #cv2.imwrite("/media/diskF/lar/code/song/img/bs_exval_114/heatmap/heatmap{}-score-{}-lab-{}.jpg".format(ii,cla_np/3,lab_np), x1)#热度图
        #cv2.imwrite("/media/diskF/lar/code/song/img/img_org/84/img{}.jpg".format(ii), img)
        if heatmap == True:
          #cv2.imwrite("/media/diskF/lar/code/song/img/bs_exval_114/img_rec/img_rec{}-score-{}-lab-{}.jpg".format(ii,cla_np/3,lab_np), img_rec)#热度
          #cv2.imwrite("/media/diskF/lar/code/song/img/bs_val_84/image/img{}-score-{}-lab-{}.png".format(ii,cla_np/3,lab_np), img,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])#原图
          cv2.imwrite("/media/diskF/lar/code/song/img/img/img{}-score-{}-lab-{}.tiff".format(ii,cla_np/3,lab_np), img, ((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
                                                                                                                                  int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
                                                                                                                                  int(cv2.IMWRITE_TIFF_XDPI), 600,
                                                                                                                                  int(cv2.IMWRITE_TIFF_YDPI), 600)))
        #cv2.imwrite("/media/diskF/lar/code/song/img/bs_exval_114/cam/cam{}-score-{}-lab-{}.jpg".format(ii,cla_np/3,lab_np), fusionimg)#2第几张图片，预测概率和标签值保存
        ##########################
        
    model.train()
 
    #cla
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    sens_c = cm_value[0][0]/(cm_value[0][0] + cm_value[0][1])
    spec_c = cm_value[1][1]/(cm_value[1][1] + cm_value[1][0])    
    
    AUC = roc_auc_score(lab, score)
    if heatmap :
      return AUC, accuracy, sens_c, spec_c, cla.value()[0],rec.value()[0],score,lab
    else :
      return AUC, accuracy, sens_c, spec_c, cla.value()[0],rec.value()[0],score,lab


def exval(socers):
    """
    计算模型在验证集上的准确率等信息
    return: 分割损失，重分割损失，重建损失，一致性损失
    """
    exval_num = np.array(pd.read_csv('/media/diskF/lar/data/lab_87.csv', encoding="GB2312"))
    j = 0
    pre=np.ones(87)
    a = 0
    nums = exval_num[:, 3].tolist()
    label = exval_num[:,1].tolist()
    for num in nums:
        socer = 0
        for i in range(0, num):
            socer = socer + socers[j]
            j = j + 1
        socer = socer / num
        pre[a] = socer
        a = a+1

    auc = roc_auc_score(label, pre)
    confusion = confusion_matrix(label, np.around(pre, 0).astype(int))
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc = (TP + TN) / float(TP + TN + FP + FN)
    sen = TP / float(TP + FN)
    spe = TN / float(TN + FP)
    label = np.array(label)
    return auc, acc, sen, spe, pre, label


def val_print(**kwargs):
    opt._parse(kwargs)
    model = getattr(models, opt.model)().eval()

    #模型所在目录/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/008-[1, 0.6]--0.0001--16/0022--0.9719--90.1361--0.7855--72.6190.pth'
    #/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/001-[1, 0]--0.0001--16/0027--0.9449--71.7460--0.6042--57.7778.pth
    #/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/008-[1, 0]--0.0001--16/0023--0.9762--90.4762--0.8135--66.6667.pth
    #/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/009-[1, 0.6]--0.0001--16/0006--0.7729--68.7075--0.7626--70.2381.pth
    #/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/008-[1, 0]--0.0001--16/0024--0.9633--89.7959--0.7740--66.6667.pth
    #model_path = '/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/009-[1, 0.6]--0.0001--16/0006--0.7729--68.7075--0.7626--70.2381.pth'
    #"0080--1.0000--100.0000--0.7460--65.4762.pth/0097--1.0000--99.6599--0.7849--67.8571.pth"
    model_path = "/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/0604-[1, 0]--0.0001--16/0080--1.0000--100.0000--0.7460--65.4762.pth"

    model.load(model_path)
    model.to(opt.device)
    model = nn.DataParallel(model, device_ids=opt.device_ids)
    
    pat_train_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, test=True)
    pat_train_dataloader = DataLoader(pat_train_data, 3, shuffle=False, num_workers=opt.num_workers)

    #pat_val_data = dataset_npy(opt.val_data_root, opt.val_data_label, is_transform=False, val=True)
    pat_val_data = dataset_npy(opt.exval_data_root, opt.exval_data_label, is_transform=False, exval=True)
    pat_val_dataloader = DataLoader(pat_val_data, 1, shuffle=False, num_workers=opt.num_workers)


    #tr_auc, tr_acc, tr_sens, tr_spec, tr_cla, tr_rec, tr_score, tr_lab = val(model, pat_train_dataloader)
    va_auc, va_acc, va_sens, va_spec, va_cla, va_rec, va_score, va_lab = val(model, pat_val_dataloader,heatmap=False)
    va_auc, va_acc, va_sens, va_spec, va_score, va_lab = exval(va_score)
    #print(va_auc, va_acc, va_sens, va_spec)
    #print(va_score.shape,va_lab.shape)
    #print([va_score, va_lab])
    np.savetxt('./result20230201/bt/checkpoints_' + name + '/MDenseNet_multi'+'/87_exval80.csv',
                   np.array(list(zip(va_score, va_lab))), delimiter=',')
    #np.savetxt('./result20230201/bt/checkpoints_' + name + '/MDenseNet_multi'+'/84_val.csv',
     #              np.concatenate([tr_score, tr_lab], axis=1), delimiter=',')
                   
    #print(str(tr_auc) + ',' + str(tr_acc) + ',' + str(tr_sens) + ',' + str(tr_spec) + ',' + '\n')
    print(str(va_auc) + ',' + str(va_acc) + ',' + str(va_sens) + ',' + str(va_spec) + ',' + '\n')

@t.no_grad()  # pytorch>=0.5
def data_print(**kwargs):
    opt._parse(kwargs)
    model = getattr(models, opt.model)().eval()

    # model_path = opt.load_model_path
    model_path = '/media/diskF/lar/code/song/result20230201/bt/checkpoints_res50+ssl_20230201/MDenseNet_multi/001-[1, 0.5]--0.0001--16_/0022--0.9368--83.1746--0.7703--68.8889.pth'


    model.load(model_path)
    model.to(opt.device)
    model = nn.DataParallel(model, device_ids=opt.device_ids)

    all_data = dataset_npy(opt.train_data_root, opt.train_data_label, is_transform=False, exval=True)
    all_dataloader = DataLoader(all_data, batch_size=3, shuffle=False, num_workers=opt.num_workers)

    for ii, ((data, mask), label) in tqdm(enumerate(all_dataloader)):
        ##################################################################
        # 保存重建结果
        path = '/media/user/Disk01/guidongqi/egfr_vgg16_0923/data0923/'

        test_keys = list(np.load(path + 'fold_1.npy')) \
                    + list(np.load(path + 'fold_2.npy')) \
                    + list(np.load(path + 'fold_3.npy')) \
                    + list(np.load(path + 'fold_4.npy')) \
                    + list(np.load(path + 'fold_5.npy'))


        cla_output, re_output, _ = model(data)
        # seg_out = output.cuda().data.cpu().numpy()
        rec_out = re_output.cuda().data.cpu().numpy()
        input = data.cuda().data.cpu().numpy()
        # mask = mask.cuda().data.cpu().numpy()
        ##########################################################保存路径
        save_path = '/media/user/Disk01/guidongqi/egfr_vgg16_0923/recons_pic/{}/'.format(opt.ex_num)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.imsave(save_path + test_keys[ii]+ '_rec.png', rec_out[1, 0])
        # plt.imsave(save_path + test_keys[ii]+ '_seg.png', seg_out[1, 0])
        plt.imsave(save_path + test_keys[ii]+ '.png', input[1, 0])
        # plt.imsave(save_path + test_keys[ii]+ '_gt.png', mask[1, 0])
        ##################################################################
        #print(dice_numpy())



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




