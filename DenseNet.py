import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import reduce
from operator import __add__


# ref: https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
    

class DenseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DenseConv, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            Conv2dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            )
        
    def forward(self, x):
        return self.block(x)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.block = nn.Sequential(
            DenseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.block(x)
    

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        out_channels = growth_rate * 4
        self.block = nn.Sequential(
                DenseConv(in_channels, out_channels, kernel_size=1),
                DenseConv(out_channels, growth_rate, kernel_size=3)
        )

    def forward(self, x):
        return torch.cat((self.block(x), x), 1)



class DenseNet(nn.Module):
    def __init__(self, in_channels, num_classes, reduction, growth_rate=12, block_config=[6, 12, 64, 48]):
        super(DenseNet, self).__init__()
        self.initial_block = nn.Sequential(
            DenseConv(in_channels, in_channels, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        layers = list()
        current_feature_channel_size = in_channels
        for block_repeat in block_config:
            block, num_channels = self._create_blocks(current_feature_channel_size, block_repeat, growth_rate)
            current_feature_channel_size = int(math.floor(num_channels*reduction))
            layers.append(block), layers.append(TransitionBlock(num_channels, current_feature_channel_size))
        self.dense_net = nn.Sequential(*layers)

        if num_classes == 2:
            self.fully_connected = nn.Linear(current_feature_channel_size, 1)
            self.act = nn.Sigmoid()
        else:
            self.fully_connected = nn.Linear(current_feature_channel_size, num_classes)
            self.act = nn.Softmax(dim=1)

    def _create_blocks(self, in_channels, num_blocks, growth_rate):
        layers = list()
        current_feature_channel_size = in_channels
        for _ in range(num_blocks):
            layers.append(DenseBlock(current_feature_channel_size, growth_rate))
            current_feature_channel_size += growth_rate
        return nn.Sequential(*layers), current_feature_channel_size
    
    def forward(self, x):
        x =  self.initial_block(x)
        x = self.dense_net(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2) # Global Average Pooling
        return self.act(self.fully_connected(x))
    

rand_tensor = torch.rand((1, 3, 224, 224))
dense_net = DenseNet(3, 2, 0.5)
print(dense_net(rand_tensor))