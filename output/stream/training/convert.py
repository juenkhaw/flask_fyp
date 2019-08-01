import torch

p = torch.load('flow_pretrain_conv3.pth.tar', map_location={'cuda:3':'cuda:0'})

new_p = {}
new_p['args'] = p['args']
new_p['epoch'] = p['train']['epoch'] + 1
new_p['state'] = {'model':p['train']['state_dict'], 'optimizer':p['train']['opt_dict'],'scheduler':p['train']['sch_dict']}
new_p['output'] = {'train_loss':p['train']['losses']['train'], 'val_loss':p['train']['losses']['val'], 
                     'train_acc':p['train']['accuracy']['train'], 'val_acc':p['train']['accuracy']['val'], 
                     'lr_decay':[], 'last_lr':1e-6, 'val_result':{'pred':None,'score':None}}
new_p['elapsed'] = {'total':p['train']['train_elapsed'], 'actual':p['train']['actual_elapsed']}

torch.save(new_p, 'flow_pretrain_conv3_2.pth.tar')