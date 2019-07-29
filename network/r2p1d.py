# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:21:55 2019

@author: Juen
"""
import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict

from network.module import Conv3D, TemplateNetwork

class SpatioTemporalConv(nn.Module):
    """
    Module consisting of spatial and temporal convolution over input volume
    
    Constructor requires:
        in_planes : channels of input volume
        out_planes : channels of output activations
        kernel_size : filter sizes (t, h, w)
        stride : (t, h, w) striding over the input volume
        bn_mom : BN momentum hyperparameter
        bn_eps : BN epsilon hyperparameter
        inter_planes : channels of the intermediate convolution (computes with formula if set to None)
        bn_relu_first_conv : applies BN and activation on spatial conv or not
        bn_relu_second_conv : applies BN and activation on temporal conv or not
        padding : [SAME/VALID] padding technique to be applied
        use_BN : applies Batch Normalization or not
        name : module name
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, stride = (1, 1, 1), 
                 bn_mom = 0.1, bn_eps = 1e-3, inter_planes = None, 
                 bn_relu_first_conv = True, bn_relu_second_conv = False, 
                 padding = 'SAME', use_bias = False, name = 'SpatioTemporalConv'):
        
        super(SpatioTemporalConv, self).__init__()
        
        # decomposing parameters into spatial and temporal components respectively
        spatial_f = [1, kernel_size[1], kernel_size[2]]
        spatial_s = [1, stride[1], stride[2]]
        spatial_p = [0, padding[1], padding[2]]
        
        temporal_f = [kernel_size[0], 1, 1]
        temporal_s = [stride[0], 1, 1]
        temporal_p = [padding[0], 0, 0]
        
        # compute the intermediate planes between spatial and temporal conv if not explicitly provided
        if inter_planes is None:
            inter_planes = int((kernel_size[0] * in_planes * out_planes * kernel_size[1] * kernel_size[2]) / 
                               (in_planes * kernel_size[1] * kernel_size[2] + kernel_size[0] * out_planes))
        
        # perform 2D Conv --> BN --> ReLU --> 1D Conv
        self.spatial_conv = Conv3D(in_planes, inter_planes, kernel_size = spatial_f, 
                                          stride = spatial_s, padding = padding, 
                                          use_BN = bn_relu_first_conv, bn_mom = bn_mom, bn_eps = bn_eps, 
                                          activation = bn_relu_first_conv, use_bias = use_bias, 
                                          name = 'spatial_')
        
        self.temporal_conv = Conv3D(inter_planes, out_planes, kernel_size = temporal_f, 
                                           stride = temporal_s, padding = padding, use_BN = bn_relu_second_conv, 
                                           bn_mom = bn_mom, bn_eps = bn_eps, 
                                           activation = bn_relu_second_conv, use_bias = use_bias, 
                                           name = 'temporal_')
        
    def forward(self, x):
        x = self.spatial_conv(x)
        #print(x.shape)
        x = self.temporal_conv(x)
        #print(x.shape)
        
        return x
    
class SpatioTemporalResBlock(nn.Module):
    """
    Residual block consists of multiple modules of spatiotemporal convolution
    
    Constructor requires:
        in_planes : channels of input volume
        out_planes : channels of output activations
        kernel_size : filter sizes (t, h, w)
        downsample : activates to apply 1*1*1 conv on shortcut to obtain volumes same as the residual
        bn_mom : BN momentum hyperparameter
        bn_eps : BN epsilon hyperparameter
        name : module name
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, downsample = False, 
                 bn_eps = 1e-3, bn_mom = 0.1, name = 'SpatioTemporalResBlock'):
        
        super(SpatioTemporalResBlock, self).__init__()
        
        self._downsample = downsample
        self.res_block = nn.Sequential(OrderedDict([]))
        self.relu = nn.Sequential(OrderedDict([]))
        
        if self._downsample:
            self.downsample_block = nn.Sequential(OrderedDict([
                
                #downsample x to be the same spatiotemporal dimension as output
                (name + '_downsample_1x1x1_conv', 
                 nn.Conv3d(in_planes, out_planes, [1, 1, 1], stride = [2, 2, 2], bias = False)),
                (name + '_downsample_bn',
                 nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps))
                ]))
                     
                    #downsample the residual
            self.res_block.add_module(name + '_conv1',
                SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME', 
                                   stride = (2, 2, 2), bn_mom = bn_mom, bn_eps = bn_eps))

        else:
            #do not downsample the x
            self.res_block.add_module(name + '_conv1', 
                SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME', 
                                   bn_mom = bn_mom, bn_eps = bn_eps))
        
        self.res_block.add_module(name + '_bn1', nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps))
        self.res_block.add_module(name + '_relu1', nn.ReLU())
        
        self.res_block.add_module(name + '_conv2', SpatioTemporalConv(out_planes, out_planes, kernel_size, padding = 'SAME', 
                                        bn_mom = bn_mom, bn_eps = bn_eps))
        self.res_block.add_module(name + '_bn2', nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps))
        
        self.relu.add_module(name + '_relu_out', nn.ReLU())
        
    def forward(self, x):
        
        res = self.res_block(x)
        
        if self._downsample:
            x = self.downsample_block(x)
            
        #print(x.shape)
        return self.relu(x + res)
    
class SpatioTemporalResModule(nn.Module):
    """
    Residual module consists of multiple residual blocks based on the network depth
    
    Constructor requires:
        in_planes : channels of input volume
        out_planes : channels of output activations
        kernel_size : filter sizes (t, h, w)
        layer_size : repetation count of residual blocks
        block_type : type of residual block
        downsample : activates to apply 1*1*1 conv on shortcut to obtain volumes same as the residual
        name : module name
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, layer_size, downsample = False, 
                 bn_eps = 1e-3, bn_mom = 0.1, name = 'SpatioTemporalResModule'):
        
        super(SpatioTemporalResModule, self).__init__()
        
        #implement the first conv to produce activation with volume same as the output
        module_name = name[:-1] + '1'
        self.res_module = nn.Sequential(OrderedDict([
                (module_name, 
                 SpatioTemporalResBlock(in_planes, out_planes, kernel_size, downsample, 
                            bn_mom = bn_mom, bn_eps = bn_eps, name = module_name))
                ]))
        
        #the rest conv operations are identical
        for i in range(layer_size - 1):
            module_name = name[:-1] + str(i + 2)
            self.res_module.add_module(module_name, 
                            SpatioTemporalResBlock(out_planes, out_planes, kernel_size, 
                                       bn_mom = bn_mom, bn_eps = bn_eps, name = module_name))
            
    def forward(self, x):
        for block in self.res_module:
            x = block(x)
            
        return x
    
class R2P1D18Net(TemplateNetwork):
    
    def __init__(self, device, num_classes, in_channel, endpoint = ['Softmax'], dropout = 0):
        
        # initializing template network
        super(R2P1D18Net, self).__init__(device, num_classes, in_channel, endpoint, dropout)
        
        # adding module block
        self.add_module('Conv1', SpatioTemporalConv(in_channel, 64, kernel_size = (3, 7, 7), 
                       stride = (1, 2, 2), padding = 'SAME', inter_planes = 45, name = 'Conv1'))
        self.add_module('Conv2_x', SpatioTemporalResModule(64, 64, kernel_size = (3, 3, 3), 
                       layer_size = 2, downsample = False, name = 'Conv2_x'))
        self.add_module('Conv3_x', SpatioTemporalResModule(64, 128, kernel_size = (3, 3, 3), 
                       layer_size = 2, downsample = True, name = 'Conv3_x'))
        self.add_module('Conv4_x', SpatioTemporalResModule(128, 256, kernel_size = (3, 3, 3), 
                       layer_size = 2, downsample = True, name = 'Conv4_x'))
        self.add_module('Conv5_x', SpatioTemporalResModule(256, 512, kernel_size = (3, 3, 3), 
                       layer_size = 2, downsample = True, name = 'Conv5_x'))
        self.add_module('Avgpool', nn.AdaptiveAvgPool3d(1))
        self.add_inter_process('Linear', {torch.Tensor.view : {'size' : (-1, 512)}})
        self.add_module('Linear', nn.Linear(512, num_classes))
        self.add_module('Dropout', nn.Dropout(p = dropout))
        self.add_module('Softmax', nn.Softmax(dim = 1))
        
        self.compile_module()
        
class R2P1D34Net(TemplateNetwork):
    
    def __init__(self, device, num_classes, in_channel, endpoint = ['Softmax'], dropout = 0):
        
        # initializing template network
        super(R2P1D34Net, self).__init__(device, num_classes, in_channel, endpoint, dropout)
        
        # adding module block
        self.add_module('Conv1', SpatioTemporalConv(in_channel, 64, kernel_size = (3, 7, 7), 
                       stride = (1, 2, 2), padding = 'SAME', inter_planes = 45, name = 'Conv1', 
                       bn_relu_second_conv=True))
        self.add_module('Conv2_x', SpatioTemporalResModule(64, 64, kernel_size = (3, 3, 3), 
                       layer_size = 3, downsample = False, name = 'Conv2_x'))
        self.add_module('Conv3_x', SpatioTemporalResModule(64, 128, kernel_size = (3, 3, 3), 
                       layer_size = 4, downsample = True, name = 'Conv3_x'))
        self.add_module('Conv4_x', SpatioTemporalResModule(128, 256, kernel_size = (3, 3, 3), 
                       layer_size = 6, downsample = True, name = 'Conv4_x'))
        self.add_module('Conv5_x', SpatioTemporalResModule(256, 512, kernel_size = (3, 3, 3), 
                       layer_size = 3, downsample = True, name = 'Conv5_x'))
        self.add_module('Avgpool', nn.AdaptiveAvgPool3d(1))
        self.add_inter_process('Linear', {torch.Tensor.view : {'size' : (-1, 512)}})
        self.add_module('Linear', nn.Linear(512, num_classes))
        self.add_module('Dropout', nn.Dropout(p = dropout))
        self.add_module('Softmax', nn.Softmax(dim = 1))
        
        self.compile_module()
    
if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = R2P1D34Net(device, 101, 3, endpoint=['Linear', 'Softmax'])
    #x = torch.randn((1, 2, 8, 112, 112)).to(device)
    #y = model.forward(x)
    
    #model.net.load_state_dict(torch.load('../pretrained/kinetic-s1m-d34-l32.pth.tar')['state_dict'])