import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import os
import importlib

from __init__ import BASE_CONF
    
#device = torch.device('cuda:0')
#exec(open('r2p1d.py').read())
#model = globals()['R2P1D18Net'](101, device, 3)

class StreamTrainer(object):

    state = {}
    args = {}
    model = None
    
    def __init__(self, form_dict):
        super(StreamTrainer, self).__init__()
        
        self.state['INITIAL'] = True
        self.args = form_dict
        
        device = torch.device('cuda:0')
        net_config = BASE_CONF['network'][self.args['network']]
        
        module = importlib.import_module('network.'+net_config['module'])
        model = getattr(module, net_config['class'])(BASE_CONF['dataset'][self.args['dataset']]['label_num'], 
                       device, BASE_CONF['channel'][self.args['modality']]).to(device)
        
        #exec(open(net_config['file']).read())
        #net = locals()[net_config['class']]
        #self.model = net(101, device, 3).to(device)