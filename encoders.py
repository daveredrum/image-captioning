import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as torchmodels
import torch.nn.functional as F
from torch.nn import Parameter

###################################################################
#                                                                 #
#                                                                 #
#                    model zoo for encoder                        #
#                                                                 #
#                                                                 #
###################################################################


__all__ = [
    # VGG
    'VGG16BN', 'EncoderVGG16BN', 'AttentionEncoderVGG16BN',
    # ResNet
    'ResNet152', 'EncoderResNet152', 'AttentionEncoderResNet152'
]


'''
VGG
'''
# feature extractor
class VGG16BN(nn.Module):
    def __init__(self):
        super(VGG16BN, self).__init__()
        vgg16 = torchmodels.vgg16_bn(pretrained=True)
        self.vgg16 = nn.Sequential(
            *list(vgg16.features.children())[:-1],
        )


    def forward(self, inputs):
        '''
        original_features: (batch_size, 512 * 14 * 14)
        '''
        original_features = self.vgg16(inputs)
        original_features = original_features.view(original_features.size(0), -1)

        return original_features

# encoder for vinila model
class EncoderVGG16BN(nn.Module):
    def __init__(self):
        super(EncoderVGG16BN, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.output_layer = nn.Sequential(
            *list(torchmodels.vgg16_bn(pretrained=True).classifier.children())[:-1],
            nn.Linear(4096, 512),
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(512, momentum=0.01)
        )
        
    
    def forward(self, inputs):
        '''
        original_features: (batch_size, 512, 14, 14)
        outputs: (batch_size, 512)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.view(inputs.size(0), 512, 14, 14)
        outputs = self.max_pool(original_features).view(batch_size, -1)
        outputs = self.output_layer(outputs)

        return outputs

# for attention
class AttentionEncoderVGG16BN(nn.Module):
    def __init__(self):
        super(AttentionEncoderVGG16BN, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=14, stride=14)
        self.global_mapping = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(512, momentum=0.01)
        )
        self.area_mapping = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(512, momentum=0.01)
        )


    def forward(self, inputs):
        '''
        original_features: (batch_size, 512, 14, 14)
        global_features: (batch_size, 512)
        area_features: (batch_size, 512, 196)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.view(inputs.size(0), 512, 14, 14)
        # (batch_size, 512, 196)
        area_features = original_features.view(batch_size, 512, -1).transpose(2, 1).contiguous()
        area_features = self.area_mapping(area_features).transpose(2, 1).contiguous().view(batch_size, 512, -1)
        # (batch_size, 512)
        global_features = self.avg_pool(original_features).view(batch_size, 512)
        global_features = self.global_mapping(global_features)

        return original_features, global_features, area_features


'''
ResNet
'''
# feature extractor
class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152, self).__init__()
        resnet = torchmodels.resnet152(pretrained=True)
        self.resnet = nn.Sequential(
            *list(resnet.children())[:-2],
        )


    def forward(self, inputs):
        '''
        original_features: (batch_size, 2048 * 7 * 7)
        '''
        original_features = self.resnet(inputs)
        original_features = original_features.view(original_features.size(0), -1)

        return original_features

# encoder for vinila model
class EncoderResNet152(nn.Module):
    def __init__(self):
        super(EncoderResNet152, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.output_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(512, momentum=0.01)
        )
        
    
    def forward(self, inputs):
        '''
        original_features: (batch_size, 2048, 7, 7)
        outputs: (batch_size, 512)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.view(inputs.size(0), 2048, 7, 7)
        outputs = self.avg_pool(original_features).view(batch_size, -1)
        outputs = self.output_layer(outputs)
        
        return outputs

# for attention
class AttentionEncoderResNet152(nn.Module):
    def __init__(self):
        super(AttentionEncoderResNet152, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.global_mapping = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(512, momentum=0.01)
        )
        self.area_mapping = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
        )
        self.area_bn = nn.BatchNorm2d(512, momentum=0.01)


    def forward(self, inputs):
        '''
        original_features: (batch_size, 2048, 7, 7)
        global_features: (batch_size, 512)
        area_features: (batch_size, 512, 49)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.view(inputs.size(0), 2048, 7, 7)
        # (batch_size, 512, 49)
        area_features = original_features.permute(0, 2, 3, 1).contiguous()
        area_features = self.area_mapping(area_features).permute(0, 3, 1, 2 ).contiguous()
        area_features = self.area_bn(area_features).view(batch_size, 512, -1)
        # (batch_size, 512)
        global_features = self.avg_pool(original_features).view(batch_size, 2048)
        global_features = self.global_mapping(global_features)

        return original_features, global_features, area_features