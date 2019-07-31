import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import importlib
from time import time
from math import ceil
import numpy as np

from dataloader import Videoset, generate_subbatches
#from __init__ import BASE_CONFIG
BASE_CONFIG = {
"channel": {
        "rgb" : 3,
        "flow" : 2
},
"network":
    {
        "r2p1d-18":
                {
                "module":"r2p1d",
                "class":"R2P1D18Net"
                },
        "r2p1d-34":
                {
                "module":"r2p1d",
                "class":"R2P1D34Net"
                },
        "i3d":
                {
                "module":"i3d",
                "class":"InceptionI3D"
                }
    },
"dataset":
    {
        "UCF-101":
                {
                "label_num" : 101,
                "base_path" : "C:\\Users\\Juen\\Desktop\\Gabumon\\Blackhole\\UTAR\\Subjects\\FYP\\dataset\\UCF-101",
                "split" : 3,
                "label_index_txt" : "classInd.txt",
                "train_txt" : ["ucf_trainlist01.txt", "ucf_trainlist02.txt", "ucf_trainlist03.txt"],
                "val_txt" : ["ucf_validationlist01.txt", "ucf_validationlist02.txt", "ucf_validationlist03.txt"],
                "test_txt" : ["ucf_testlist01.txt", "ucf_testlist02.txt", "ucf_testlist03.txt"]
                },
        "HMDB-51":
                {
                "label_num" : 51,
                "base_path" : "C:\\Users\\Juen\\Desktop\\Gabumon\\Blackhole\\UTAR\\Subjects\\FYP\\dataset\\HMDB-51",
                "split" : 3,
                "label_index_txt" : "classInd.txt", 
                "train_txt" : ["hmdb_trainlist01.txt", "hmdb_trainlist02.txt", "hmdb_trainlist03.txt"],
                "val_txt" : [], 
                "test_txt" : ["hmdb_testlist01.txt", "hmdb_testlist02.txt", "hmdb_testlist03.txt"]
                }
    }
}

class StreamTrainer(object):

    update = {'progress' : False, 'result' : False}
    args = {}
    model = None
    main_output = None
    
    def __init__(self, form_dict):
        super(StreamTrainer, self).__init__()
        self.args = form_dict
        
    def init_network(self):
        """
        FOR ALL TASK
        Initializes network modules and standing by in the computing device
        """
        # define device
        self.device = torch.device(self.args['device'])
        net_config = BASE_CONFIG['network'][self.args['network']]
        
        # define network model
        print('train_stream/initializing network')
        module = importlib.import_module('network.'+net_config['module'])
        self.model = getattr(module, net_config['class'])(self.device, BASE_CONFIG['dataset'][self.args['dataset']]['label_num'], 
                       BASE_CONFIG['channel'][self.args['modality']], endpoint=['Linear', 'Softmax'])
        self.model.compile_module()
        
    def load_pretrained_model(self):
        """
        FOR TRAINING ONLY (transfer learning and training from scratch)
        Loads a pretrained model (if any) into the network
        """
        # apply pretrained model if there is any
        if self.args['pretrain_model'] != 'none':
            print('train_stream/loading pretrained model')
            key_error = self.model.net.load_state_dict(torch.load('pretrained/'+self.args['pretrain_model']), strict = False)
            if self.args['freeze_point'] != None:
                self.model.freeze(self.args['freeze_point'])
            print('train_stream/WARNING missing/unexpected keys in loading state', key_error)
            
    def load_resume_model(self):
        """
        FOR RESUMING TRAINING ONLY
        Loads a halfway trained model (including state_dict of optimizer and scheduler)
        """
        pass
    
    def load_evaluating_model(self):
        """
        FOR TESTING ONLY
        Loads a maturely trained model for evaluation
        """
        pass
    
    def load_train_val_dataset(self):
        """
        FOR TRAINING and RESUME TRAINING
        Prepares loader for training and validation set
        """   
        # prepare dataloaders for training and validation set
        print('train_stream/preparing dataloaders')
        train_dataloader = DataLoader(Videoset(self.args, 'train'), batch_size=self.args['batch_size'], shuffle=True)
        if BASE_CONFIG['dataset'][self.args['dataset']]['val_txt'] != []:
            val_dataloader = DataLoader(Videoset(self.args, 'val'), batch_size=self.args['val_batch_size'], shuffle=False)
            self.dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        else:
            self.dataloaders = {'train': train_dataloader}
        
    def load_test_dataset(self):
        """
        FOR TESTING ONLY
        Prepares loader for testing set
        """
        print('train_stream/preparing dataloaders')
        self.dataloaders = DataLoader(Videoset(self.args, 'test'), batch_size=self.args['batch_size'], shuffle=False)
        
    def setup_training(self):
        
        self.init_network()
        self.load_pretrained_model()
        self.load_train_val_dataset()
                
        # define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.net.parameters(), lr = self.args['base_lr'], momentum = self.args['momentum'], 
                                   weight_decay = self.args['l2decay'])
        
        # define LR scheduler
        if self.args['lr_scheduler'] == 'none':
            self.scheduler = None
        elif self.args['lr_scheduler'] == 'stepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.args['step_size'], gamma = self.args['lr_reduce_ratio'], 
                                                       last_epoch = self.args['last_step'])
        elif self.args['lr_scheduler'] == 'dynamic':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.args['patience'], 
                                                                  threshold=self.args['loss_threshold'], min_lr=self.args['min_lr'])
        
        torch.cuda.empty_cache()
        print('train_stream/setup completed for TRAINING')
        
    def setup_resume_training(self):
        pass
    
    def setup_testing(self):
        pass
    
    def save_main_output(self, path, **contents):
        self.main_output = contents
        if self.args['output_name'] != '':
            torch.save(self.main_output, path+self.args['output_name']+'.pth.tar')
            
    def compute_confusion_matrix(self, result, dataloader):
        """
        result: expected to be a softmax output tensor
        dataloader: for the frequency table
        """
        result = result.cpu().detach().numpy()
        freq = dataloader.dataset._Y_freq
        size = (len(freq), BASE_CONFIG['dataset'][self.args['dataset']]['label_num'])
        val_result = {'score': np.empty(size), 'pred':np.empty(size)}
        
        count = 0
        for i, f in enumerate(freq):
            # based on predicted label
            pred = np.argmax(result[count:count + f], axis = 1)
            val_result['pred'][i] = np.bincount(pred, minlength=size[1]) / f * 100
            
            # based on softmax scores
            val_result['score'][i] = np.sum(result[count:count+f], axis=0) / f
            
            count += f
            
        return val_result
        
    def train_net(self): # TO BE FUSED WITH RESUME TRAINING TASK
        
        print('train_stream/initiate TRAINING')
        
        subbatch_sizes = {'train' : self.args['sub_batch_size'], 'val' : self.args['val_batch_size']}
        subbatch_count = {'train' : self.args['batch_size'], 'val' : self.args['val_batch_size']}
        performances = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'lr_decay':[], 'last_lr':None, 'val_result':None}
        epoch = 1
        elapsed = {'total':0, 'train':0}
        timer = {'total':0, 'train':0}
        
        for epoch in range(epoch, self.args['epoch'] + 1):
            timer['total'] = time()
            
            for phase in self.dataloaders.keys():
                torch.cuda.empty_cache()
                
                # anticipating total batch count
                batch = 1
                self.total_batch = int(ceil(len(self.dataloaders[phase].dataset) / subbatch_count[phase]))
                
                # reset the loss and accuracy
                current_loss = 0
                current_correct = 0
                
                # put the model into appropriate mode
                if phase == 'train':
                    self.model.net.train()
                else:
                    self.model.net.eval()
                
                val_epoch_results = torch.tensor([], dtype = torch.float).to(self.device)
                
                # for each mini batch of dataset
                for inputs, labels in self.dataloaders[phase]:
                    torch.cuda.empty_cache()
                    
                    print('Phase', phase, '| Current batch', str(batch), '/', str(self.total_batch), end = '\n')
                    batch += 1
                    self.update['progress'] = True
                    
                    # place the input and label into memory of gatherer unit
                    inputs = inputs.to(self.device)
                    labels = labels.long().to(self.device)
                    self.optimizer.zero_grad()
            
                    # partioning each batch into subbatches to fit into memory
                    if phase == 'train':
                        sub_inputs, sub_labels = generate_subbatches(subbatch_sizes[phase], inputs, labels)
                    else:
                        sub_inputs = [inputs]
                        sub_labels = [labels]
                        
                    # with enabling gradient descent on parameters during training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # initialize empty tensor for validation result
                        outputs = torch.tensor([], dtype = torch.float).to(self.device)
                        sb = 0
                        
                        for sb in range(len(sub_inputs)):
                            torch.cuda.empty_cache()
                            
                            # this is where actual training starts
                            timer['train'] = time()
                                
                            output = self.model.forward(sub_inputs[sb])
                            
                            if phase == 'train':
                                # transforming outcome from a series of scores to a single scalar index
                                # indicating the index where the highest score held
                                _, preds = torch.max(output['Softmax'], 1)
                                
                                # compute the loss
                                loss = self.criterion(output['Linear'], sub_labels[sb])
                                
                                # accumulate gradients on parameters
                                loss.backward()
                                
                                # accumulate loss and true positive for the current subbatch
                                current_loss += loss.item() * sub_inputs[sb].size(0)
                                current_correct += torch.sum(preds == sub_labels[sb].data)
                                
                            else:
                                # append the validation result until all subbatches are tested on
                                outputs = torch.cat((outputs, output['Softmax']))
                                
                            # this is where actual training ends
                            elapsed['train'] += time() - timer['train']
                            
                    # avearging over validation results and compute val loss
                    if phase == 'val':
                        
                        current_loss += self.criterion(outputs, labels).item() * labels.shape[0]
                        
                        _, preds = torch.max(outputs, 1)
                        current_correct += torch.sum(preds == labels.data)
                        
                        val_epoch_results = torch.cat((val_epoch_results, outputs))
            
                    # update parameters
                    if phase == 'train':
                        self.optimizer.step()
                        
                # compute the loss and accuracy for the current batch
                epoch_loss = current_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = float(current_correct) / len(self.dataloaders[phase].dataset)
                
                performances[phase+'_loss'].append(epoch_loss)
                performances[phase+'_acc'].append(epoch_acc)
                
                # step on reduceLRonPlateau with val acc
                if self.scheduler is not None:
                    if 'val' in self.dataloaders.keys():
                        if phase == 'val':
                            self.scheduler.step(epoch_acc)
                    else:
                        self.scheduler.step(epoch_acc)
                        
                    # get current learning rate
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                        
                    # update lr decay schedule
                    if lr != performances['last_lr']:
                        performances['last_lr'] = lr
                        performances['lr_decay'].append(epoch)
                        
                # getting confusion matrix on validation result
                if phase == 'val':
                    performances['val_result'] = self.compute_confusion_matrix(val_epoch_results, self.dataloaders[phase])
            
            elapsed['total'] += time() - timer['total']
            print('train_stream/currently completed epoch', epoch)
            
            # save all outputs
            self.save_main_output('output/stream/training/', args = self.args, epoch = epoch, state = {
                    'model':self.model.net.state_dict(), 'optimizer':self.optimizer.state_dict(), 
                    'scheduler': None if self.scheduler == None else self.scheduler.state_dict()
                    }, output = performances, elapsed = elapsed)
        
if __name__ == '__main__':
    from json import loads
    j = """{
      "base_lr": 0.01, 
      "batch_size": 32, 
      "clip_len": 8, 
      "crop_h": 112, 
      "crop_w": 112, 
      "dataset": "UCF-101", 
      "debug_mode": "distributed", 
      "debug_train_size": 2, 
      "debug_val_size": 4, 
      "device": "cuda:0", 
      "dropout": 0.0, 
      "epoch": 3, 
      "freeze_point": "none", 
      "is_batch_size": false, 
      "is_debug_mode": true, 
      "is_mean_sub": false, 
      "is_rand_flip": true, 
      "l2decay": 0.01, 
      "last_step": -1, 
      "loss_threshold": 0.0001, 
      "lr_reduce_ratio": 0.1, 
      "lr_scheduler": "dynamic", 
      "min_lr": 0.0, 
      "modality": "rgb", 
      "momentum": 0.1, 
      "network": "r2p1d-18", 
      "output_compare": [], 
      "output_name": "test", 
      "patience": 10, 
      "pretrain_model": "none", 
      "resize_h": 128, 
      "resize_w": 171, 
      "split": 1, 
      "step_size": 10, 
      "sub_batch_size": 2, 
      "val_batch_size": 14,
      "test_method": "none"
    }"""
    temp = StreamTrainer(loads(j))
    temp.setup_training()
    temp.train_net()