import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DrConv(nn.Module):
    def __init__(self, in_channel, out_channel=1, kernel_length=3, direction=0, padding='same', stride=1,bias=True):
        super().__init__()
        self.kernel_size = kernel_length
        self.padding = padding
        self.stride = stride
        self.w = nn.Parameter(torch.randn(out_channel, in_channel, 1, kernel_length), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(out_channel),requires_grad=True)if bias else None
        self.reshaper=None
        if direction == 1:
            self.w = nn.Parameter(torch.randn(out_channel, in_channel, kernel_length,1), requires_grad=True)
        elif direction == 2:
            self.reshaper = torch.eye(kernel_length,requires_grad=False)
        elif direction == 3:
            self.reshaper = torch.flip(torch.eye(kernel_length,requires_grad=False),[-1])

        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        kernel = self.w
        if self.reshaper is not None:
            kernel = kernel * self.reshaper
        kernel = kernel.to(x.device)
        bias = self.b.to(x.device)
        return F.conv2d(x,kernel,stride=self.stride,padding=self.padding,bias=bias)


if __name__ == '__main__':
    from torch import nn
    from torch.nn import functional as F
    import torch
    import math

    net=nn.Sequential(
        DrConv(in_channel=3, out_channel=16, kernel_length=3, direction=0),
        DrConv(in_channel=16,out_channel=8, kernel_length=5, direction=1),
        DrConv(in_channel=8,out_channel=16, kernel_length=3, direction=2),
        DrConv(in_channel=16,out_channel=3, kernel_length=5, direction=3)
    )
    lq = torch.randn(1, 3, 512, 512).cuda()
    gt = torch.randn(1, 3, 512, 512).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters())
    for i in range(1024):
        pred = net(lq)
        optimizer.zero_grad()
        loss = criterion(pred, gt)
        loss.backward()
        print(f'{i}: {loss.item()}')
        optimizer.step()


