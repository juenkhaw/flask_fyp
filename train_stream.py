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

    state = {'INIT' : False}
    args = {}
    model = None
    
    def __init__(self):
        super(StreamTrainer, self).__init__()
        
    def init(self, form_dict):
        
        self.args = form_dict
        
        device = torch.device('cuda:0')
        net_config = BASE_CONF['network'][self.args['network']]
        
        module = importlib.import_module('network.'+net_config['module'])
        self.model = getattr(module, net_config['class'])(BASE_CONF['dataset'][self.args['dataset']]['label_num'], 
                       device, BASE_CONF['channel'][self.args['modality']]).to(device)
        
        self.state['INIT'] = True