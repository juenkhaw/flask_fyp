import torch

p = torch.load('rgb_pretrain_conv2.pth.tar')
a = p['args']

args = {'base_lr':a.lr, 'batch_size':a.batch_size, 'clip_len':a.clip_length, 'crop_h':112, 'crop_w':112, 
        'dataset':'UCF-101', 'debug_mode':'none', 'debug_train_size': 32, 'debug_val_size': 32, 'device':'cuda:0', 
        'dropout': 0, 'epoch': 50, 
        'freeze_point': 'Conv2_x', #
        'is_batch_size': False, 'is_debug_mode': False, 'is_mean_sub': False, 'is_rand_flip':False, 
        'l2decay': a.l2wd, 'last_step': -1, 'loss_threshold': 0.0001, 'lr_reduce_ratio': 0.1, 
        "lr_scheduler": "dynamic", 
        "min_lr": 0.000001, #
        "modality": "rgb", #
        "momentum": a.momentum, #
        "network": "r2p1d-34", 
        "output_compare": [], 
        "output_name": "rgb_pretrain_conv2", #
        "patience": 10, 
        "pretrain_model": "kinetic-s1m-d34-l32.pth.tar", #
        "resize_h": 128, 
        "resize_w": 171, 
        "split": 1, 
        "step_size": 10, 
        "sub_batch_size": 4, 
        "test_method": "none", 
        "val_batch_size": 20
        }

p['args'] = args
torch.save(p, 'rgb_pretrain_conv2.pth.tar')