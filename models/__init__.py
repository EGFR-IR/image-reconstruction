#from .unet.unet_torch_new import UNet
from .fpn.FPN import FPN
from .fpn.FPN_2 import FPN2
from .fpn.FPN_3 import FPN3
from .fpn.FPN_4 import FPN4
from .fpn.Alexnet import Alexnet
from .fpn.Alexnet_2 import Alexnet2
from .resnet.resnet50_rec_new import resnet50
from .resnet.resnet50_rec_new_1228 import resnet50_1228
from .resnet.resnet50_rec_new_0717 import resnet50_0717

from .resnet.resnet50_rec_new_0819 import resnet50_0819
# from .resnet.resnet50_rec import resnet50
from .resnet.resnet50_unrec import resnet50_2
from .vgg.vgg import vgg16, vgg16_bn

from .multi_model.vgg import VGG16
from .multi_model.densenet import MDenseNet
from .multi_model.inceptionv3 import InceptionV3
from .multi_model.mobilenetv3 import MobileNetV3
from .multi_model.cascnn import cascnn
from .multi_model.tian import DenseNet_v2