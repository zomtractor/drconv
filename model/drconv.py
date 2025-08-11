import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DrConv(nn.Module):
    def __init__(self,in_channel,out_channel=1,kernel_size=3,padding='same',stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.conv_h = nn.Conv2d(in_channel, out_channel, (1, kernel_size), padding=padding, bias=False,stride=stride)
        self.conv_v = nn.Conv2d(in_channel, out_channel, (kernel_size, 1), padding=padding, bias=False,stride=stride)
        self.conv_d1 = torch.nn.Parameter(
            torch.randn(out_channel,in_channel,1, kernel_size), requires_grad=True
        )
        self.conv_d2 = torch.nn.Parameter(
            torch.randn(out_channel,in_channel,1, kernel_size), requires_grad=True
        )
        nn.init.kaiming_uniform_(self.conv_d1)
        nn.init.kaiming_uniform_(self.conv_d2)
        self.eyes = torch.eye(kernel_size,requires_grad=False)
        self.reyes = torch.flip(self.eyes,[-1])

    def forward(self, x):
        h = self.conv_h(x)
        v = self.conv_v(x)
        d1 = F.conv2d(x,self.conv_d1*self.eyes,stride=self.stride,padding=self.padding)
        d2 = F.conv2d(x,self.conv_d2*self.reyes,stride=self.stride,padding=self.padding)
        return h, v, d1, d2


