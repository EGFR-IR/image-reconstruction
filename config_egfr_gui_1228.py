# -*- coding: utf-8 -*
import warnings
import torch as t


class DefaultConfig(object):
    env = 'gdq-main'  # visdom 环境
    vis_port = 8097   # visdom 端口
    model = 'MDenseNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致MDenseNet resnet50_1228
    ex_num = '001+[1,0.6]'

    train_data_root = '/media/diskF/lar/data/split_600/npy_crop_600/image/'  # 训练集存放路径
    train_data_label = '/media/diskF/lar/data/split_600/npy_crop_600/mask/'

    # train_data_root = '/media/user/Disk01/guidongqi/egfr_vgg16_0923/data0923/npy_crop_224/image/'  # 训练集存放路径
    # train_data_label = '/media/user/Disk01/guidongqi/egfr_vgg16_0923/data0923/npy_crop_224/mask/'
    val_data_root = train_data_root  # 验证集存放路径
    val_data_label = train_data_label
    exval_data_root = '/media/diskF/lar/data/data_95/image/'  # 验证集存放路径
    exval_data_label = '/media/diskF/lar/data/data_95/mask/'
    #exval_data_root = train_data_root  
    #exval_data_label = train_data_label
    test_data_root = train_data_root  # 测试集存放路径
    test_data_label = train_data_label


    # 加载预训练的模型的路径，为None代表不加载
    load_model_path = None
    # load_model_path = '/media/user/Disk01/guidongqi/egfr_vgg16_0923/checkpoints_rec_2_1026/resnet50_multi/r_42513_1026_1--5e-05--16/0034--0.4967--0.4846--0.8691--0.8328.pth'

    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    # result_file = 'result.csv'

    max_epoch = 100
    lr = 0.0001# initial learning rate
    lr_decay = 0.5 # when val_loss increase, lr = lr*lr_decay  # 5231-4; 0.0005-0.1 
    
    
    
    # weight_decay = 0e-5  # 损失函数
    device_ids = [0,1]

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
