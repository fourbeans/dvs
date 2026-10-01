import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.auto_cuda import neuron_kernel as ac_neuron_kernel
# from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from retina_model import Retina

from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class TEBN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        self.p = nn.Parameter(torch.ones(4, 1, 1, 1, 1))
    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y

class TEBNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        # x = self.act(x)
        return y
class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        # x = self.act(x)
        return x
    
class PMSNaa(nn.Module):
    def __init__(self, T: int, neuron_size, surrogate_function: surrogate.SurrogateFunctionBase, m_weight=1):
        super().__init__()
        self.surrogate_function = surrogate_function
        # self.read = nn.Linear(neuron_size, neuron_size)
        self.neuron_m = nn.Parameter(torch.zeros((T, neuron_size)), requires_grad=True)
        self.memory_loss = nn.Parameter(torch.zeros(T, neuron_size), requires_grad=False)
        self.T = T
        self.neuron_size = neuron_size
        self.m_weight = m_weight
        print('use pmsn')
        
        order = 2
        self.order = order
        weight = torch.ones([order])
        for i in range(order - 2, -1, -1):
            weight[i] = weight[i + 1] / 2.
        self.weight = nn.Parameter(weight)
        self.threshold = nn.Parameter(torch.as_tensor(-1.))
        
    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]
        return weight

    def forward(self, x_seq: torch.Tensor):
        # memory
        weight = self.gen_gemm_weight(self.T)
        simil = (weight @ self.neuron_m).unsqueeze(1)
        if len(x_seq.shape) == 5:
            x_seq = x_seq.permute(1,0,2,3,4)
            # t = x_seq.shape[0]
            # b = x_seq.shape[1]
            # c = x_seq.shape[2]
            # x_seq = self.m_weight  * simil.unsqueeze(3).unsqueeze(4).repeat((1,1,1,x_seq.shape[-2], x_seq.shape[-1])) + x_seq
        elif len(x_seq.shape) == 3:
            x_seq = x_seq.permute(1,0,2)
            # t = x_seq.shape[0]
            # b = x_seq.shape[1]
            # c = x_seq.shape[2]
            # x_seq = self.m_weight * simil + x_seq
            
        h_seq = F.linear(x_seq.transpose(0, -1), weight, self.threshold)
        spike = self.surrogate_function(h_seq)
        # if self.training:
        #     self.memory_loss.data = (x_seq + h_seq.view(x_seq.shape)).reshape(t,b,c,-1).mean(1).mean(
        #         -1)
        if len(x_seq.shape) == 5:
            x_seq = x_seq.permute(1,0,2,3,4)
        elif len(x_seq.shape) == 3:
            x_seq = x_seq.permute(1,0,2)
        return spike.view(x_seq.shape)

class PMSN(nn.Module):

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight


    def __init__(self, order: int = 2, surrogate_function = surrogate.ATan(), exp_init: bool=True, neuron_size=64, T=4, m_weight=0.1):
        super().__init__()
        self.neuron_m = nn.Parameter(torch.zeros((T, neuron_size)), requires_grad=True)
        self.memory_loss = nn.Parameter(torch.zeros(T, neuron_size), requires_grad=False)
        self.m_weight = m_weight
        
        self.order = order
        if exp_init:
            weight = torch.ones([order])
            for i in range(order - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.

            self.weight = nn.Parameter(weight)
        else:
            self.weight = torch.ones([1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight[0])
        self.threshold = nn.Parameter(torch.as_tensor(-1.))
        self.surrogate_function = surrogate_function


    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [N, T, *]
        b = x_seq.shape[0]
        t = x_seq.shape[1]
        c = x_seq.shape[2]
        simil = self.neuron_m.unsqueeze(0)
        if len(x_seq.shape) == 5:
            x_seq = self.m_weight  * simil.unsqueeze(3).unsqueeze(4).repeat((1, 1, 1, x_seq.shape[-2], x_seq.shape[-1])) + x_seq
        elif len(x_seq.shape) == 3:
            x_seq = self.m_weight * simil + x_seq
        
        weight = self.gen_gemm_weight(x_seq.shape[1])
        h_seq = F.linear(x_seq.transpose(1, -1), weight, self.threshold)
        h_seq = h_seq.transpose(1, -1)
        
        if self.training:
            self.memory_loss.data = (x_seq + h_seq.view(x_seq.shape)).reshape(b,t,c,-1).mean(0).mean(
                -1)
        return self.surrogate_function(h_seq)


# sj
class ReLUX(nn.Module):
    def __init__(self, thre=4):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)

relu4 = ReLUX(thre=4)

class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=4):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None

class Multispike(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs) / 4
    
class PMSNSJ(nn.Module):

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight

    def __init__(self, order: int = 2, surrogate_function = surrogate.ATan(), exp_init: bool=True, neuron_size=64, T=4, m_weight=0.1):
        super().__init__()
        self.multispike = Multispike()  # 使用 Multispike 实例
        self.fc = nn.Linear(T, T)
        self.neuron_m = nn.Parameter(torch.zeros((T, neuron_size)), requires_grad=True)
        self.memory_loss = nn.Parameter(torch.zeros(T, neuron_size), requires_grad=False)
        self.T = T
        self.neuron_size = neuron_size
        self.m_weight = m_weight

        self.order = order
        if exp_init:
            weight = torch.ones([order])
            for i in range(order - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.

            self.weight = nn.Parameter(weight)
        else:
            self.weight = torch.ones([1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight[0])
        self.threshold = nn.Parameter(torch.as_tensor(-1.))
        self.surrogate_function = surrogate_function
        
        print('use pmsnSJ')

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [N, T, *]
        b = x_seq.shape[0]
        t = x_seq.shape[1]
        c = x_seq.shape[2]
        simil = self.neuron_m.unsqueeze(0)
        if len(x_seq.shape) == 5:
            x_seq = self.m_weight  * simil.unsqueeze(3).unsqueeze(4).repeat((1, 1, 1, x_seq.shape[-2], x_seq.shape[-1])) + x_seq
        elif len(x_seq.shape) == 3:
            x_seq = self.m_weight * simil + x_seq
        
        weight = self.gen_gemm_weight(x_seq.shape[1])
        h_seq = F.linear(x_seq.transpose(1, -1), weight, self.threshold)
        spike = self.multispike(h_seq.transpose(1, -1))
        if self.training:
            self.memory_loss.data = (x_seq + h_seq.view(x_seq.shape)).reshape(t, b, c, -1).mean(1).mean(-1)

        # 返回一个有相同数据但不同维度的Tensor
        return spike.view(x_seq.shape)
 
class VGGPMSN_reg(nn.Module):
    def __init__(self, tau=0.5, T=4):
        super(VGGPMSN_reg, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=64),
            Layer(64, 128, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=128),
            pool,
            Layer(128, 256, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=256),
            Layer(256, 256, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=256),
            pool,
            Layer(256, 512, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=512),
            Layer(512, 512, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=512),
            pool,
            Layer(512, 512, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=512),
            Layer(512, 512, 3, 1, 1),
            PMSNSJ(T=T, neuron_size=512),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # input = add_dimention(input, self.T)
        
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, PMSNSJ):
                features.append(x.mean(1).mean(-1).mean(-1))
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        features.append(x.mean(1))
        return x, features