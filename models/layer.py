import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################################################################
# F blocks
class Normalizer(nn.Module):
    def __init__(self, numChannels, momentum=0.985, affine=True, channelNorm=True, device='cpu'):
        super(Normalizer, self).__init__()

        self.momentum = momentum
        self.numChannels = numChannels
        self.affine = affine
        self.channelNorm = channelNorm

        self.movingAverage = torch.zeros(1, numChannels, 1).to(device)
        self.movingVariance = torch.ones(1, numChannels, 1).to(device)

        if affine:
            self.BatchNormScale = nn.Parameter(torch.ones(1, numChannels, 1)).to(device)
            self.BatchNormBias = nn.Parameter(torch.zeros(1, numChannels, 1)).to(device)

    def forward(self, x):

        # Apply channel wise normalization
        if self.channelNorm:
            
            x = (x-torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 0.00001)
            
        # If in training mode, update moving per channel statistics
        if self.training:
            
            newMean = torch.mean(x, dim=2, keepdim=True)
            self.movingAverage = ((self.momentum * self.movingAverage) + ((1 - self.momentum) * newMean)).detach()
            x = x - self.movingAverage

            newVariance = torch.mean(torch.pow(x, 2), dim=2, keepdim=True)
            self.movingVariance = ((self.momentum * self.movingVariance) + ((1 - self.momentum) * newVariance)).detach()
            x = x / (torch.sqrt(self.movingVariance) + 0.00001)
            
        else:

            ma = self.movingAverage[:x.size(0),:,:]
            mv = self.movingVariance[:x.size(0),:,:]
            x = (x - ma) / (torch.sqrt(mv) + 0.00001)
            
        # Apply batch norm affine transform
        if self.affine:
            
            x = x * torch.abs(self.BatchNormScale)
            x = x + self.BatchNormBias
            
        return x

class SeperableDenseNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernelSize,
                 groups=1, dilation=1, channelNorm=True, down_sampling=0, device='cpu'):
        super(SeperableDenseNetUnit, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernelSize = kernelSize
        self.groups = groups
        self.dilation = dilation
        self.down_sampling = down_sampling
        # Convolutional transforms
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernelSize,
                               padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) // 2, dilation=dilation).to(device)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=4*out_channels, groups=1, kernel_size=1,
                               padding=0, dilation=1).to(device)

        self.conv3 = nn.Conv1d(in_channels=4*out_channels, out_channels=4*out_channels, groups=4*out_channels, kernel_size=kernelSize,
                               padding=(kernelSize + ((kernelSize - 1) * (dilation - 1)) - 1) // 2, dilation=dilation).to(device)
        self.conv4 = nn.Conv1d(in_channels=4*out_channels, out_channels=out_channels, groups=1, kernel_size=1,
                               padding=0, dilation=1).to(device)

        self.conv1 = nn.utils.weight_norm(self.conv1, 'weight').to(device)
        self.conv2 = nn.utils.weight_norm(self.conv2, 'weight').to(device)
        self.conv3 = nn.utils.weight_norm(self.conv3, 'weight').to(device)
        self.conv4 = nn.utils.weight_norm(self.conv4, 'weight').to(device)

        self.norm1 = Normalizer(numChannels=4 * out_channels, channelNorm=channelNorm, device=device)
        self.norm2 = Normalizer(numChannels=out_channels, channelNorm=channelNorm, device=device)

    def forward(self, x):
        # Apply first convolution block
        y = self.conv2(self.conv1(x))
        
        y = self.norm1(y)
        
        y = F.selu(y)
        
        # Apply second convolution block
        y = self.conv4(self.conv3(y))
        
        y = self.norm2(y)
        
        y = F.selu(y)
        
        y = torch.cat((y, x), dim=1)

        # Apply Down sampling if down_sampling != 0
        if self.down_sampling:
            y = F.max_pool1d(y, kernel_size=self.down_sampling)

        # Return densely connected feature map
        return y

########################################################################################################################
# Define the Sleep model

class SkipLSTM(nn.Module):
    def __init__(self, in_channels, out_channels=4, hiddenSize=32, is_construct=False, device='cpu'):
        super(SkipLSTM, self).__init__()

        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Bidirectional LSTM to apply temporally across input channels
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=hiddenSize, num_layers=1, batch_first=True, dropout=0.0,
                           bidirectional=True).to(device)
        self.rnn = nn.utils.weight_norm(self.rnn, name='weight_ih_l0').to(device)
        self.rnn = nn.utils.weight_norm(self.rnn, name='weight_hh_l0').to(device)

        # Output convolution to map the LSTM hidden states from forward and backward pass to the output shape
        self.outputConv1 = nn.Conv1d(in_channels=hiddenSize*2, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0).to(device)
        self.outputConv1 = nn.utils.weight_norm(self.outputConv1, name='weight').to(device)

        self.outputConv2 = nn.Conv1d(in_channels=hiddenSize, out_channels=out_channels, groups=1, kernel_size=1, padding=0).to(device)
        self.outputConv2 = nn.utils.weight_norm(self.outputConv2, name='weight').to(device)

        # Residual mapping
        self.identMap1 = nn.Conv1d(in_channels=in_channels, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0).to(device)

        # Construct mode set
        self.is_construct = is_construct

    def print(self, line):
        if self.is_construct: print(line)

    def forward(self, x):

        self.print(f"> [Inside skipLSTM Layer]")
        self.print(f"> Input x     : {x.size()}")   # torch.Size([2, 348, 7200])
        y = x.permute(0, 2, 1)
        self.print(f"> permuted1 y : {y.size()}")   # torch.Size([2, 7200, 348])
        y, z = self.rnn(y)
        self.print(f"> after RNN y : {y.size()}")   # torch.Size([2, 7200, 256])
        z = None
        y = y.permute(0, 2, 1)
        self.print(f"> permuted2 y : {y.size()}")   # torch.Size([2, 256, 7200])
        y = torch.tanh((self.outputConv1(y) + self.identMap1(x)) / 1.41421)
        self.print(f"> conv1+ident : {y.size()}")   # torch.Size([2, 128, 7200])
        y = self.outputConv2(y)
        self.print(f"> outputConv2 : {y.size()}\n") # torch.Size([2, 4, 7200])
        return y 