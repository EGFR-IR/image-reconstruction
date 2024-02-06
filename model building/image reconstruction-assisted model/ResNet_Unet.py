# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
'''
class double_conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return 
'''
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(out_ch )
        )
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv(x) + self.shortcut(x))
        #return nn.ReLU(inplace=True)(self.conv(x))

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

'''
        self.dconv_last=nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64,3,1)           
        )
'''



class resnet34_Unet(nn.Module):

    def __init__(self,  num_block=2):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(BasicBlock, 64, 3, 1)
        self.conv3_x = self._make_layer(BasicBlock, 128, 4, 2)
        self.conv4_x = self._make_layer(BasicBlock, 256, 6, 2)
        self.conv5_x = self._make_layer(BasicBlock, 512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Sequential(    #   
                #nn.Linear(1024, 1024),
                #nn.ReLU(True),
                #nn.Dropout(0.4)  ,            
                nn.Linear(512, 256)  ,
                nn.Sigmoid(),
                nn.Dropout(0.4),
                nn.Linear(256, 2)
                )
        
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)#7->14
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)#14->28
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)#28->56
        self.dconv_up1 = double_conv(64 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)#56->112
        self.dconv_last = double_conv(64+64, 64)
        self.up = nn.ConvTranspose2d(64, 64, 2, stride=2)#112->224
        self.recon = nn.Conv2d(64, 3, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)                      #[16, 64, 112, 112]
        temp=self.maxpool(conv1)
        conv2 = self.conv2_x(temp)                  #[16, 64, 56, 56]
        conv3 = self.conv3_x(conv2)                #16, 128, 28, 28
        conv4 = self.conv4_x(conv3)                #16, 256, 14, 14
        bottle = self.conv5_x(conv4)                #[16, 512, 7, 7]
        output = self.avg_pool(bottle)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        #x = self.upsample(bottle)
        x = self.up3(bottle)
        #print(x.shape)#16, 512, 14, 14]
        #print(conv4.shape)#16, 256, 14, 14
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up3(x)
        #x = self.upsample(x)
        x = self.up2(x)
        #print(x.shape)#16, 256, 28, 28
        #print(conv3.shape)#16, 128, 28, 28
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up2(x)
        #x = self.upsample(x)
        x = self.up1(x)
        #print(x.shape)#[16, 128, 56, 56]
        #print(conv2.shape)#16, 64, 56, 56
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up1(x)
        x = self.up0(x)
        #x = self.upsample(x)
        #print(x.shape)#16, 64, 112, 112
        #print(conv1.shape)#16, 64, 112, 112
        x = torch.cat([x,conv1],dim=1)
        #print(x.shape)#16, 128, 112, 112
        out = self.dconv_last(x)
        #print(out.shape)
        out = self.up(out)
        #print(out.shape)
        out = self.recon(out)
        #print(out.shape)
        out = nn.Sigmoid()(out)
        #print(out.shape)#16,3,224,224
        #print(x.shape, conv1.shape, conv2.shape, conv3.shape, bottle.shape)
        return output,out


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


#def resnet34_Unet():
#    """ return a ResNet 34 object
#    """
#    model=ResNet(BasicBlock, [3, 4, 6, 3])
#    return model


if __name__ == '__main__':
    net = resnet34_Unet()
    print(net)
    if isinstance(net,torch.nn.DataParallel):
		    net = net.module
    for k in net.state_dict():
        print(k)
    print(net)
    x = torch.rand((16, 3, 224, 224))
    net.forward(x)


