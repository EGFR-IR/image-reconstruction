import warnings
import torch as t


class DefaultConfig(object):
    env = 'gdq-main'  
    vis_port = 8097  
    model = 'MDenseNet' 
    ex_num = '001+[1,0.6]'

    train_data_root = '/media/diskF/data/split_600/npy_crop_600/image/'  
    train_data_label = '/media/diskF/data/split_600/npy_crop_600/mask/'

    val_data_root = train_data_root 
    val_data_label = train_data_label
    exval_data_root = '/media/diskF/lar/data/data_95/image/' 
    exval_data_label = '/media/diskF/lar/data/data_95/mask/'
    test_data_root = train_data_root  
    test_data_label = train_data_label


   load_model_path = None

    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb

    max_epoch = 100
    lr = 0.0001# initial learning rate
    lr_decay = 0.5 # when val_loss increase, lr = lr*lr_decay  # 5231-4; 0.0005-0.1 

    device_ids = [0,1]

    def _parse(self, kwargs):
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
