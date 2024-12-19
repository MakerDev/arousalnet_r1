
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .layer import Normalizer, SeperableDenseNetUnit, SkipLSTM

class ArousalWithStageApnea(nn.Module):
    def __init__(self, sfreq=100, num_signals=12, kernel_size=25, channel_multiplier=2, is_construct=False, device='cpu'):
        super(ArousalWithStageApnea, self).__init__()
        
        # Config 
        self.sfreq              = sfreq
        self.num_signals        = num_signals
        self.kernel_size        = kernel_size
        self.channel_multiplier = channel_multiplier
        
        self.is_construct = is_construct
        self.device = device
        # Module params 
        # >DCU1
        ds_kernels = self.__calc_ds_kernels(self.sfreq)

        # >DCU2
        depth_dcu2 = 11
        dcu2_const_inch = len(ds_kernels)
        dilations = self.__calc_dilations(depth_dcu2)

        # >skipLSTM 
        lstm_const_inch = len(ds_kernels)+len(dilations)

        # Module Initialization
        self.dcu1 = nn.Sequential(*(
            SeperableDenseNetUnit(
                in_channels=((i*self.channel_multiplier)+1)*self.num_signals, 
                out_channels=self.channel_multiplier*self.num_signals,
                kernelSize=(2*self.kernel_size)+1, 
                groups=1, dilation=1, channelNorm=False, 
                down_sampling=ds_kernel, device=self.device)
            for i, ds_kernel in enumerate(ds_kernels) ))

        self.dcu2 = nn.Sequential(*(
            SeperableDenseNetUnit(
                in_channels=(((i+dcu2_const_inch)*self.channel_multiplier)+1)*self.num_signals, 
                out_channels=self.channel_multiplier*self.num_signals,
                kernelSize=self.kernel_size, 
                groups=1, dilation=dilation, channelNorm=True, 
                down_sampling=0, device=self.device)
            for i, dilation in enumerate(dilations) ))
        
        self.skipLSTM = SkipLSTM(
            in_channels=(lstm_const_inch*self.channel_multiplier+1)*self.num_signals, 
            hiddenSize=self.channel_multiplier*64, 
            out_channels=4, 
            is_construct=is_construct, device=self.device)
            
    # def to(self, device=None):
    #     for module in self.dcu1: module.to(device)
    #     for module in self.dcu2: module.to(device)
    #     self.skipLSTM.rnn = self.skipLSTM.rnn.to(device)
    #     return super(ArousalWithStageApnea, self).to(device)

    def marginalize(self, x):

        self.print(f"> [marginalize Layer]")
        self.print(f"> Input tensor : {x.size()}")
        p_joint = F.log_softmax(x, dim=1)
        self.print(f"> p_joint      : {p_joint.size()}\n")

        # Compute marginal for arousal predictions
        self.print(f"> [Compute marginal for arousal predictions]")
        p_arousal = p_joint[::, 3, ::]
        self.print(f"> p_arousal : {p_arousal.size()}")
        x1 = torch.cat((torch.log(1 - torch.exp(p_arousal.unsqueeze(1))), p_arousal.unsqueeze(1)), dim=1)
        self.print(f"> x1        : {x1.size()}\n")

        # Compute marginal for apnea predictions
        self.print(f"> [Compute marginal for apnea predictions]")
        p_apnea = p_joint[::, 1, ::]
        self.print(f"> p_apnea   : {p_arousal.size()}")
        x2 = torch.cat((torch.log(1 - torch.exp(p_apnea.unsqueeze(1))), p_apnea.unsqueeze(1)), dim=1)
        self.print(f"> x2        : {x2.size()}\n")

        # Compute marginal for sleep/wake predictions
        self.print(f"> [Compute marginal for sleep/wake predictions]")
        p_wake = p_joint[::, 0, ::]
        self.print(f"> p_wake    : {p_arousal.size()}")
        x3 = torch.cat((p_wake.unsqueeze(1), torch.log(1 - torch.exp(p_wake.unsqueeze(1)))), dim=1)
        self.print(f"> x3        : {x3.size()}\n")

        return x1, x2, x3

    def print(self,line):
        if self.is_construct: print(line)

    def forward(self, x):

        self.print(f"[Input Layer]")
        x = x.detach().contiguous()
        self.print(f"Input tensor     : {x.size()}\n")

        # Downsampling to 1 entity per second
        self.print(f"[DCU1 Module]")
        x = self.dcu1(x)
        self.print(f"after DCU1   : {x.size()}\n")
        
        self.print(f"[DCU2 Module]")
        x = self.dcu2(x)
        self.print(f"after DCU2   : {x.size()}\n")
        
        # Bidirectional skip LSTM and convert joint predictions to marginal predictions
        self.print(f"[skipLSTM Layer]")
        x = self.skipLSTM(x)
        self.print(f"after skipLSTM : {x.size()}")
        x1, x2, x3 = self.marginalize(x)
        self.print(f"after marginalize : {x1.size()}, {x2.size()}, {x3.size()}\n")
        
        if not(self.training):
            x1 = torch.exp(x1)
            x2 = torch.exp(x2)
            x3 = torch.exp(x3)

        return (x1, x2, x3)

    @staticmethod
    def __calc_ds_kernels(n):

        # Calc Primes 
        a = [False,False] + [True]*(n-1)
        primes=[]

        for i in range(2,n+1):
            if a[i]:
                primes.append(i)
                for j in range(2*i, n+1, i):
                    a[j] = False

        # Calc Factors
        factors = []
        for p in primes:
            count = 0
            while n % p == 0:
                n /= p
                count += 1
            if count > 0:
                factors.append((p, count))

        factors = [f[0] for f in factors for i in range(f[1])]   

        return factors 

    @staticmethod
    def __calc_dilations(dense_depth):
        dilations = list(map(lambda i: np.power(2,i), range(0, dense_depth//2)))
        dilations += list(map(lambda i: np.power(2,i), range(dense_depth//2, -1, -1)))
        return dilations