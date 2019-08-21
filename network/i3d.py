# -*- coding: utf-8 -*-	
"""	
Created on Sat Jan 26 21:56:29 2019	
@author: Juen	
"""	

import torch	
import torch.nn as nn	

from network.module import Conv3D, MaxPool3DSame, TemplateNetwork

class InceptionModule(nn.Module):	

    #out_planes expected to have 6 numbers	
    #corresponding to each Convs	
    def __init__(self, in_planes, out_planes):	

        super(InceptionModule, self).__init__()	
        
        self.branch0 = Conv3D(in_planes, out_planes[0], kernel_size = (1, 1, 1), name = '')	
        self.branch1a = Conv3D(in_planes, out_planes[1], kernel_size = (1, 1, 1), name = '')	
        self.branch1b = Conv3D(out_planes[1], out_planes[2], kernel_size = (3, 3, 3), name = '')	
        self.branch2a = Conv3D(in_planes, out_planes[3], kernel_size = (1, 1, 1), name = '')	
        self.branch2b = Conv3D(out_planes[3], out_planes[4], kernel_size = (3, 3, 3), name = '')	
        self.branch3a = MaxPool3DSame(kernel_size = (3, 3, 3), stride = (1, 1, 1))	
        self.branch3b = Conv3D(in_planes, out_planes[5], kernel_size = (1, 1, 1), name = '')	

    def forward(self, x):	

        x0 = self.branch0(x)	
        x1 = self.branch1b(self.branch1a(x))	
        x2 = self.branch2b(self.branch2a(x))	
        x3 = self.branch3b(self.branch3a(x))	

         #concatenate the activations from each branch at the channel dimension	
        return torch.cat([x0, x1, x2, x3], dim = 1)
    
class InceptionI3D(TemplateNetwork):
    
    def __init__(self, device, num_classes, in_channel, endpoint = ['Softmax'], dropout = 0):
        
        # initializing template network
        super(InceptionI3D, self).__init__(device, num_classes, in_channel, endpoint, dropout)
        
        # adding module block
        self.add_module('Conv3d_1a_7x7', Conv3D(in_channel, 64, kernel_size = (7, 7, 7), 	
                       stride = (2, 2, 2), padding = 'SAME'))
        self.add_module('MaxPool3d_2a_3x3', MaxPool3DSame(kernel_size = (1, 3, 3), 	
                       stride = (1, 2, 2), padding = (0, 0, 0)))
        self.add_module('Conv3d_2b_1x1', Conv3D(64, 64, kernel_size = (1, 1, 1)))
        self.add_module('Conv3d_2c_3x3', Conv3D(64, 192, kernel_size = (3, 3, 3), 	
                       padding = 'SAME'))
        self.add_module('MaxPool3d_3a_3x3', MaxPool3DSame(kernel_size = (1, 3, 3), 	
                       stride = (1, 2, 2), padding = (0, 0, 0)))
        self.add_module('Mixed_3b', InceptionModule(192, [64, 96, 128, 16, 32, 32]))
        self.add_module('Mixed_3c', InceptionModule(256, [128, 128, 192, 32, 96, 64]))
        self.add_module('MaxPool3d_4a_3x3', MaxPool3DSame(kernel_size = (3, 3, 3), 	
                       stride = (2, 2, 2), padding = (0, 0, 0)))
        self.add_module('Mixed_4b', InceptionModule(480, [192, 96, 208, 16, 48, 64]))
        self.add_module('Mixed_4c', InceptionModule(512, [160, 112, 224, 24, 64, 64]))
        self.add_module('Mixed_4d', InceptionModule(512, [128, 128, 256, 24, 64, 64]))
        self.add_module('Mixed_4e', InceptionModule(512, [112, 144, 288, 32, 64, 64]))
        self.add_module('Mixed_4f', InceptionModule(528, [256, 160, 320, 32, 128, 128]))
        self.add_module('MaxPool3d_5a_2x2', MaxPool3DSame(kernel_size = (2, 2, 2),
                       stride = (2, 2, 2), padding = (0, 0, 0)))
        self.add_module('Mixed_5b', InceptionModule(832, [256, 160, 320, 32, 128, 128]))
        self.add_module('Mixed_5c', InceptionModule(832, [384, 192, 384, 48, 128, 128]))
        self.add_module('Avg_pool', nn.AvgPool3d(kernel_size = (2, 7, 7), stride = (1, 1, 1)))
        self.add_module('Dropout', nn.Dropout(dropout))
        self.add_module('Logits', Conv3D(1024, num_classes, kernel_size = (1, 1, 1), 	
                                    padding = 'VALID', activation = None, 	
                                    use_BN = False, use_bias = True))
        self.add_inter_process('Linear', {torch.Tensor.view : {'size' : (-1, num_classes * 7)}})
        self.add_module('Linear', nn.Linear(num_classes * 7, num_classes))
        self.add_module('Softmax', nn.Softmax(dim = 1))
        
        # compile into a complete network
        self.compile_module()

if __name__ == '__main__':	
    device = torch.device('cuda:0')
    #model = InceptionI3D(101, device).to(device)
    model = InceptionI3D(device, 101, 3, endpoint=['Logits', 'Softmax'])
    
    missingkey = model.net.load_state_dict(torch.load('../pretrained/i3d_rgb_charades.pth.tar'), strict=False)
    
#    for name, param in model.net.named_parameters():
#        if param.requires_grad:
#            print(name, '####', param.shape)
    
#    ppp = torch.load('../../i3d/rgb_imagenet.pt')
#    keys = []
#    for k, v in oldp.items():
#        keys.append(k)
#    keys.sort()
#    for k in keys:
#        print(k, '####', oldp[k].shape)
    
    x = torch.randn((1, 3, 16, 224, 224)).to(device)
    y = model.forward(x)