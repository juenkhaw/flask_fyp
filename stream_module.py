import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import importlib
from time import time
from math import ceil
import numpy as np
from os import path, makedirs

from dataloader import Videoset, generate_subbatches
from graph import cv_confusion_matrix
from . import BASE_CONFIG

#BASE_CONFIG = {
#"channel": {
#        "rgb" : 3,
#        "flow" : 2
#},
#"network":
#    {
#        "r2p1d-18":
#                {
#                "module":"r2p1d",
#                "class":"R2P1D18Net"
#                },
#        "r2p1d-34":
#                {
#                "module":"r2p1d",
#                "class":"R2P1D34Net"
#                },
#        "i3d":
#                {
#                "module":"i3d",
#                "class":"InceptionI3D"
#                }
#    },
#"dataset":
#    {
#        "UCF-101":
#                {
#                "label_num" : 101,
#                "base_path" : "C:\\Users\\Juen\\Desktop\\Gabumon\\Blackhole\\UTAR\\Subjects\\FYP\\dataset\\UCF-101",
#                "split" : 3,
#                "label_index_txt" : "classInd.txt",
#                "train_txt" : ["ucf_trainlist01.txt", "ucf_trainlist02.txt", "ucf_trainlist03.txt"],
#                "val_txt" : ["ucf_validationlist01.txt", "ucf_validationlist02.txt", "ucf_validationlist03.txt"],
#                "test_txt" : ["ucf_testlist01.txt", "ucf_testlist02.txt", "ucf_testlist03.txt"]
#                },
#        "HMDB-51":
#                {
#                "label_num" : 51,
#                "base_path" : "C:\\Users\\Juen\\Desktop\\Gabumon\\Blackhole\\UTAR\\Subjects\\FYP\\dataset\\HMDB-51",
#                "split" : 3,
#                "label  _index_txt" : "classInd.txt", 
#                "train_txt" : ["hmdb_trainlist01.txt", "hmdb_trainlist02.txt", "hmdb_trainlist03.txt"],
#                "val_txt" : [], 
#                "test_txt" : ["hmdb_testlist01.txt", "hmdb_testlist02.txt", "hmdb_testlist03.txt"]
#                }
#    }
#}

class StreamTrainer(object):

    update = {'init': False, 'progress' : False, 'result' : False, 'complete' : False}
    batch = 0
    total_batch = 0
    phase = 'null'
    epoch = 0
    args = {}
    model = None
    main_output = None
    elapsed = {'train':0, 'total':0}
    compare = []
    
    def __init__(self, form_dict):
        super(StreamTrainer, self).__init__()
        self.args = form_dict
        
    def init_network(self, endpoint = ['Linear', 'Softmax']):
        """
        FOR ALL TASK
        Initializes network modules and standing by in the computing device
        """
        # define device
        self.device = torch.device(self.args['device'])
        net_config = BASE_CONFIG['network'][self.args['network']]
        
        # define network model
        print('stream/initializing network')
        module = importlib.import_module('network.'+net_config['module'])
        self.model = getattr(module, net_config['class'])(self.device, BASE_CONFIG['dataset'][self.args['dataset']]['label_num'], 
                       BASE_CONFIG['channel'][self.args['modality']], endpoint=endpoint)
        #self.model.compile_module()
        
    def load_pretrained_model(self):
        """
        FOR TRAINING ONLY (transfer learning and training from scratch)
        Loads a pretrained model (if any) into the network
        """
        # apply pretrained model if there is any
        if self.args['pretrain_model'] != 'none':
            print('train_stream/loading pretrained model')
            key_error = self.model.net.load_state_dict(torch.load('pretrained/'+self.args['pretrain_model'], 
                map_location=lambda storage, location: storage.cuda(int(self.args['device'][-1])) if 'cuda' in self.args['device'] else storage), strict = False)
            if self.args['freeze_point'] != 'none':
                self.model.freeze(self.args['freeze_point'])
            print('train_stream/WARNING missing/unexpected keys while loading state', key_error)
            
    def load_resume_model(self):
        """
        FOR RESUMING TRAINING ONLY
        Loads a halfway trained model (including state_dict of optimizer and scheduler)
        """
        print('resume_stream/loading previous model state')
        state_url = self.args['half_model'].split('\\')
        state_url[-2] = 'state'
        state = torch.load('\\'.join(state_url), map_location=lambda storage, location: storage.cuda(int(self.args['device'][-1])) if 'cuda' in self.args['device'] else storage)
        key_error = self.model.net.load_state_dict(state['model'], strict = False)
        if self.args['freeze_point'] != 'none':
            self.model.freeze(self.args['freeze_point'])
        print('resume_stream/WARNING missing/unexpected keys while loading state', key_error)
        return state
    
    def load_evaluating_model(self):
        """
        FOR TESTING ONLY
        Loads a maturely trained model for evaluation
        """
        print('test_stream/loading complete model state')
        state_url = self.args['full_model'].split('\\')
        state_url[-2] = 'state'
        state = torch.load('\\'.join(state_url), map_location=lambda storage, location: storage.cuda(int(self.args['device'][-1])) if 'cuda' in self.args['device'] else storage)
        key_error = self.model.net.load_state_dict(state['model'], strict = False)
        print('test_stream/WARNING missing/unexpected keys while loading state', key_error)
        del state
        torch.cuda.empty_cache()
    
    def load_train_val_dataset(self):
        """
        FOR TRAINING and RESUME TRAINING
        Prepares loader for training and validation set
        """   
        # prepare dataloaders for training and validation set
        print('stream/preparing dataloaders')
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
        self.dataloaders = DataLoader(Videoset(self.args, 'test'), batch_size=self.args['test_batch_size'], shuffle=False)
        
    def setup_LR_scheduler(self):
        # define LR scheduler
        if self.args['lr_scheduler'] == 'none':
            self.scheduler = None
        elif self.args['lr_scheduler'] == 'stepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.args['step_size'], gamma = self.args['lr_reduce_ratio'], 
                                                       last_epoch = self.args['last_step'])
        elif self.args['lr_scheduler'] == 'dynamic':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.args['patience'], 
                                                                  threshold=self.args['loss_threshold'], min_lr=self.args['min_lr'])
            
    def load_comparing_states(self):
        if len(self.args['output_compare']) != 0:
            self.compare = [torch.load(x, map_location=lambda storage, loc: storage) for x in self.args['output_compare']]
        
    def setup_training(self):
        
        self.init_network()
        self.load_pretrained_model()
        self.load_train_val_dataset()
                
        # define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.net.parameters(), lr = self.args['base_lr'], momentum = self.args['momentum'], 
                                   weight_decay = self.args['l2decay'])
        self.setup_LR_scheduler()
        
        self.load_comparing_states()
        
        torch.cuda.empty_cache()
        print('train_stream/setup completed for TRAINING')
        self.update['init'] = True
        
    def setup_resume_training(self):
        # load output file of half-model, combining argument updates from resume form
        self.main_output = torch.load(self.args['half_model'], 
                map_location=lambda storage, location: storage.cuda(int(self.args['device'])) if 'cuda' in self.args['device'] else storage)
        self.main_output['args'].update(self.args)
        self.args = self.main_output['args']
        
        #print(self.args)
        
        self.init_network()
        state = self.load_resume_model()
        self.load_train_val_dataset()
        
        # carry forward loss, previous states of optimizer and scheduler
        print('resume_stream/loading optimzer and scheduler states')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.net.parameters(), lr = self.args['base_lr'], momentum = self.args['momentum'], 
                                   weight_decay = self.args['l2decay'])
        self.optimizer.load_state_dict(state['optimizer'])
                
        self.setup_LR_scheduler()
        if self.scheduler != None:
            self.scheduler.load_state_dict(state['optimizer'])
            
        self.load_comparing_states()
            
        del state
        torch.cuda.empty_cache()
        print('resume_stream/setup completed for RESUMING TRAINER')
        self.update['init'] = True
        self.update['result'] = True
        
        if 'val' in self.dataloaders.keys():
            self.save_cfm(self.main_output['output']['val_result'], 'val', self.dataloaders['val'])
    
    def setup_testing(self):
        # load output file of completed model, combining argument updates from resume form
        self.main_output = torch.load(self.args['full_model'], 
                map_location=lambda storage, location: storage.cuda(int(self.args['device'])) if 'cuda' in self.args['device'] else storage)
        self.main_output['args'].update(self.args)
        self.args = self.main_output['args']
        
        self.init_network(endpoint=['Softmax'])
        self.load_evaluating_model()
        self.load_test_dataset()
        
        # put model into evaluation mode
        self.model.net.eval()
        
        torch.cuda.empty_cache()
        print('test_stream/setup completed for TESTING')
        self.update['init'] = True
        
    def save_cfm(self, data, mode, dataloader):
        read_path = path.join('static',self.args['dataset'],'split'+str(self.args['split']),mode)
        output_path = path.join('static', 'stream_val_result', self.args['dataset'], 'split'+str(self.args['split']), 'anonymous' if self.args['output_name'] == '' else self.args['output_name'])
        for metric in ['pred', 'score']:
            for sort in ['none', 'asc', 'desc']:
                cv_confusion_matrix(read_path, data, 
                                    dataloader.dataset._label_list, target = metric, sort = sort, 
                                    output_path = output_path)
    
    def save_main_output(self, _path, **contents):
        self.main_output = contents
        
        if not path.exists(_path):
            makedirs(_path)
        
        if self.args['output_name'] != '':
            if 'testing' in _path:
                torch.save(self.main_output, _path+self.args['output_name']+'_'+self.args['test_method']+'.pth.tar')
            else:
                torch.save(self.main_output, _path+self.args['output_name']+'.pth.tar')
            
    def compute_confusion_matrix(self, result, dataloader):
        """
        result: expected to be a softmax output tensor
        dataloader: for the frequency table
        """
        if self.phase == 'val':
            result = result.cpu().detach().numpy()
        freq = dataloader.dataset._Y_freq
        size = (len(freq), BASE_CONFIG['dataset'][self.args['dataset']]['label_num'])
        val_result = {'score': np.empty(size), 'pred':np.empty(size)}
        
        count = 0
        for i, f in enumerate(freq):
            if f == 0:
                val_result['pred'][i] = np.nan
                val_result['score'][i] = np.nan
            else:
                # based on predicted label
                pred = np.argmax(result[count:count + f], axis = 1)
                val_result['pred'][i] = np.bincount(pred, minlength=size[1]) / f
                
                # based on softmax scores
                val_result['score'][i] = np.sum(result[count:count+f], axis=0) / f
                
                count += f
            
        return val_result
        
    def train_net(self): # TO BE FUSED WITH RESUME TRAINING TASK
        
        print('stream/initiate TRAINING')
        subbatch_sizes = {'train' : self.args['sub_batch_size'], 'val' : self.args['val_batch_size']}
        subbatch_count = {'train' : self.args['batch_size'], 'val' : self.args['val_batch_size']}        
        
        if self.main_output == None:
            performances = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'lr_decay':[], 'last_lr':None, 'val_result':None}
            epoch = 1
            self.elapsed = {'total':0, 'train':0}
        else:
            performances = self.main_output['output']
            epoch = self.main_output['epoch'] + 1
            self.elapsed = self.main_output['elapsed']
            
        timer = {'total':0, 'train':0}
        
        for self.epoch in range(epoch, self.args['epoch'] + 1):
            timer['total'] = time()
            
            for self.phase in ['train', 'val']:
                torch.cuda.empty_cache()
                
                if 'val' not in self.dataloaders.keys():
                    continue
                
                # anticipating total batch count
                self.batch = 0
                self.total_batch = int(ceil(len(self.dataloaders[self.phase].dataset) / subbatch_count[self.phase]))
                
                # reset the loss and accuracy
                current_loss = 0
                current_correct = 0
                
                # put the model into appropriate mode
                if self.phase == 'train':
                    self.model.net.train()
                else:
                    self.model.net.eval()
                
                val_epoch_results = torch.tensor([], dtype = torch.float).to(self.device)
                
                # for each mini batch of dataset
                for inputs, labels in self.dataloaders[self.phase]:
                    torch.cuda.empty_cache()
                    
                    self.batch += 1
                    print('epoch', self.epoch, 'phase', self.phase, '| Current batch', str(self.batch), '/', str(self.total_batch), end = '\n')
                    self.update['progress'] = True
                    
                    # place the input and label into memory of gatherer unit
                    inputs = inputs.to(self.device)
                    labels = labels.long().to(self.device)
                    self.optimizer.zero_grad()
            
                    # partioning each batch into subbatches to fit into memory
                    if self.phase == 'train':
                        sub_inputs, sub_labels = generate_subbatches(subbatch_sizes[self.phase], inputs, labels)
                    else:
                        sub_inputs = [inputs]
                        sub_labels = [labels]
                        
                    # with enabling gradient descent on parameters during training self.phase
                    with torch.set_grad_enabled(self.phase == 'train'):
                        
                        # initialize empty tensor for validation result
                        outputs = torch.tensor([], dtype = torch.float).to(self.device)
                        outputs2 = torch.tensor([], dtype = torch.float).to(self.device)
                        sb = 0
                        
                        for sb in range(len(sub_inputs)):
                            torch.cuda.empty_cache()
                            
                            # this is where actual training starts
                            timer['train'] = time()
                                
                            output = self.model.forward(sub_inputs[sb])
                            
                            if self.phase == 'train':
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
                                outputs = torch.cat((outputs, output['Linear']))
                                outputs2 = torch.cat((outputs2, output['Softmax']))
                                
                            # this is where actual training ends
                            self.elapsed['train'] += time() - timer['train']
                            
                    # avearging over validation results and compute val loss
                    if self.phase == 'val':
                        
                        current_loss += self.criterion(outputs, labels).item() * labels.shape[0]
                        
                        _, preds = torch.max(outputs, 1)
                        current_correct += torch.sum(preds == labels.data)
                        
                        val_epoch_results = torch.cat((val_epoch_results, outputs2))
            
                    # update parameters
                    if self.phase == 'train':
                        self.optimizer.step()
                        
                    # one batch done
                    
                        
                # compute the loss and accuracy for the current batch
                epoch_loss = current_loss / len(self.dataloaders[self.phase].dataset)
                epoch_acc = float(current_correct) / len(self.dataloaders[self.phase].dataset)
                
                performances[self.phase+'_loss'].append(epoch_loss)
                performances[self.phase+'_acc'].append(epoch_acc)
                
                # step on reduceLRonPlateau with val acc
                if self.scheduler is not None:
                    if 'val' in self.dataloaders.keys():
                        if self.phase == 'val':
                            self.scheduler.step(epoch_acc)
                    else:
                        self.scheduler.step(epoch_acc)
                        
                    # get current learning rate
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                        
                    # update lr decay schedule
                    if lr != performances['last_lr']:
                        performances['last_lr'] = lr
                        performances['lr_decay'].append(self.epoch)
                        
                # getting confusion matrix on validation result
                if self.phase == 'val':
                    performances['val_result'] = self.compute_confusion_matrix(val_epoch_results, self.dataloaders[self.phase])
                    self.save_cfm(performances['val_result'], 'val', self.dataloaders['val'])
            
            self.elapsed['total'] += time() - timer['total']
            #print('train_stream/currently completed epoch', self.epoch)
            
            save_path = 'output/stream/'+self.args['dataset']+'/split'+str(self.args['split'])+'/'
            
            # save all outputs
            self.save_main_output(save_path+'training/', args = self.args, epoch = self.epoch,
                                  output = performances, elapsed = self.elapsed)
            
            if not path.exists(save_path+'state/'):
                makedirs(save_path+'state/')
                
            print('Epoch %d | lr %.1E | TrainLoss %.4f | ValLoss %.4f | TrainAcc %.4f | ValAcc %.4f' % 
              (self.epoch, lr, performances['train_loss'][self.epoch - 1], performances['val_loss'][self.epoch - 1], 
               performances['train_acc'][self.epoch - 1], performances['val_acc'][self.epoch - 1]))
            
            torch.save({
                    'model':self.model.net.state_dict(), 'optimizer':self.optimizer.state_dict(), 
                    'scheduler': None if self.scheduler == None else self.scheduler.state_dict()
                    }, save_path+'state/'+self.args['output_name']+'.pth.tar')
                    
            # one epoch done
            self.update['result'] = True
            
    def test_net(self):
        print('stream/initiate TESTING')
        
        all_scores = []
        cfm = np.zeros((len(self.dataloaders.dataset._Y_freq), len(self.dataloaders.dataset._label_list)))
        test_correct = {'top-1' : 0, 'top-5' : 0}
        test_acc = {'top-1' : 0, 'top-5' : 0}
        
        self.batch = 0
        self.total_batch = int(ceil(len(self.dataloaders.dataset) / self.dataloaders.batch_size))
        
        for inputs, labels in self.dataloaders:
        
            torch.cuda.empty_cache()
            self.batch += 1
            print('Phase test | Current batch =', str(self.batch), '/', str(self.total_batch), end = '\n')
            self.update['progress'] = True
            
            # getting dimensions of input
            batch_size = inputs.shape[0]
            clips_per_video = inputs.shape[1]
            inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], 
                                 inputs.shape[4], inputs.shape[5])
            
            # moving inputs to gatherer unit
            inputs = inputs.to(self.device)
                
            # reshaping labels tensor
            labels = np.array(labels).reshape(len(labels), 1)
            
            # partioning each batch into subbatches to fit into memory
            sub_inputs = generate_subbatches(self.args['test_subbatch_size'], inputs)
            
            del inputs
            torch.cuda.empty_cache()
            
            # with gradient disabled, perform testing on each subbatches
            with torch.set_grad_enabled(False):
            
                outputs = torch.tensor([], dtype = torch.float).to(self.device)
                sb = 0
                
                for sb in range(len(sub_inputs)):
                    
                    torch.cuda.empty_cache()
    
                    output = self.model.forward(sub_inputs[sb])
                    outputs = torch.cat((outputs, output['Softmax']))
                                        
            # detach the predicted scores         
            outputs = outputs.cpu().detach().numpy()
                
            # average the scores for each classes across all clips that belong to the same video
            averaged_score = np.average(np.array(np.split(outputs, batch_size)), axis = 1)
            
            # concet into all scores
            if all_scores == []:
                all_scores = averaged_score
            else:
                all_scores = np.concatenate((all_scores, averaged_score), axis = 0)
            
            # retrieve the label index with the top-5 scores
            top_k_indices = np.argsort(averaged_score, axis = 1)[:, ::-1][:, :5]
            
            for i, label in enumerate(labels):
                cfm[label, top_k_indices[i, 0]] += 1
            
            # compute number of matches between predicted labels and true labels
            test_correct['top-1'] += np.sum(top_k_indices[:, 0] == np.array(labels).ravel())
            test_correct['top-5'] += np.sum(top_k_indices == np.array(labels))
        
        # compute accuracy over predictions on current batch
        test_acc['top-1'] = float(test_correct['top-1']) / len(self.dataloaders.dataset)
        test_acc['top-5'] = float(test_correct['top-5']) / len(self.dataloaders.dataset)
        
        save_path = 'output/stream/'+self.args['dataset']+'/split'+str(self.args['split'])+'/'
        
        self.save_main_output(save_path+'testing/', args = self.args, acc = test_acc, pred = all_scores, 
                              result = self.compute_confusion_matrix(all_scores, self.dataloaders), cfm = cfm)
        
        self.update['result'] = True
        
if __name__ == '__main__':
    from json import loads
    j = """{
      "base_lr": 0.01, 
      "batch_size": 32,
      "clip_len" : 16,
      "crop_h": 224, 
      "crop_w": 224, 
      "dataset": "UCF-101", 
      "debug_mode": "peek", 
      "debug_train_size": 8, 
      "debug_val_size": 8, 
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
      "network": "i3d", 
      "output_compare": [], 
      "output_name": "test",
      "patience": 10, 
      "pretrain_model": "i3d_rgb_charades.pth.tar",
      "resize_h": 256, 
      "resize_w": 342, 
      "split": 1, 
      "step_size": 10, 
      "sub_batch_size": 1,
      "val_batch_size": 2,
      "test_method": "none",
      "half_model" : "flow_conv2.pth.tar",
      "full_model": "flow_conv2.pth.tar",
      "test_batch_size": 6,
      "test_subbatch_size": 60,
      "debug_test_size": 2
    }"""
    temp = StreamTrainer(loads(j))
    temp.setup_training()
    temp.train_net()
    #temp.setup_testing()
    #temp.test_net()