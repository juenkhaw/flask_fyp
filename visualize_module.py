# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:46:46 2019

@author: Juen
"""

import torch
import torch.nn as nn
import numpy as np
import cv2

import importlib

from . import BASE_CONFIG
from .dataloader import read_frames_for_visualization, transform_buffer, denormalize_buffer

class GBP(object):
    """
    Class containing guided backprop visualization tools
    
    Constructor requires:
        model : network module to be modified to perform guided backprop
    """
    
    def init_hook_input_space(self):
        """
        register a hook function at the conv module that is nearest to input space
        to retrieve the gradient map flowing to the input layer
        """
        
        def hook_first_layer(module, grad_input, grad_output):
            # expect output grads has shape of (1, 3, t, h, w)
#            for i in range(len(grad_input)):
#                if grad_input[i] is not None:
#                    print(i, grad_input[i].shape)
#            for i in range(len(grad_output)):
#                if grad_input[i] is not None:
#                    print(i, grad_output[i].shape)
            self.output_grads = grad_input[0]
            
        #first_block = self.model.net_order[0][1]

        for module in self.model.net.modules():
            if isinstance(module, nn.Conv3d):
                first_block = module
                break
            
        # assign the hook function onto the first conv module
        self.conv_hook = first_block.register_backward_hook(hook_first_layer)
        
    def init_hook_relu(self):
        """
        register hook functions in all relu modules throughout the network
        forward prop : to store the output activation map as buffer and to be used in backprop
        backward prop : to zero out the upstream gradient of dead neurons based on the corresponding 
                        activation map stored during forward prop
        """
        
        def hook_relu_forward(module, input, output):
            # store the relu output of current layer
            # to be used as mask to zero out the upstream gradient
            self.forward_relu_outputs.append(output)
            
        def hook_relu_backward(module, grad_input, grad_output):

            # create gradient mask on corresponding relu output
            # prevent gradient to flow through dead neurons
            grad_mask = (self.forward_relu_outputs.pop() > 0)
            
            # zeroing out upstream gradients with negative value
            grad = grad_input[0]
            grad[grad < 0] = 0
            
            # back prop
            grad_mask = grad_mask.type_as(grad)
            return (grad_mask * grad, )
        
        # registering relu hook functions
        self.relu_hook = []
        for module in self.model.net.modules():
            if isinstance(module, nn.ReLU):
                self.relu_hook.append(module.register_forward_hook(hook_relu_forward))
                self.relu_hook.append(module.register_backward_hook(hook_relu_backward))
    
    def __init__(self, model):
        
        self.model = model
        self.model.output_endpoints = ['Softmax']
        
        self.output_grads = None
        self.forward_relu_outputs = []
        
        self.init_hook_input_space()
        self.init_hook_relu()
        
    def de_init(self):
        self.conv_hook.remove()
        for h in self.relu_hook:
            h.remove()
        del self.conv_hook
        del self.relu_hook
        
    def compute_grad(self, input, target_class):
        """
        Feeds network with testing input volume and retrieves the input space gradient
        
        Inputs:
            input : testing volume of clip
            filter_pos : selection of a kernel activation map in the top (nearest to output) module
                
        Outputs:
            input sapce gradient map
        """
        
        self.model.net.eval()            
        self.model.net.zero_grad()
        
        # expected output has shape of (1, chnl, t, h, w)
        output = self.model.forward(input)
        
        # zero out rest of the neurons activation map

        activation = output['Softmax'][0, target_class]

        # backprop
        activation.backward()
        
        grads = self.output_grads.cpu()
        grads = transform_buffer(grads, True)
        #grads = np.uint8(((grads - np.min(grads)) / (np.max(grads) - np.min(grads))) * 255)
        
        if grads.size != input.size:
            final_grads = np.float32(np.zeros(tuple(grads.shape)[:2] + tuple(input.shape)[3:] + (input.shape[1],)))
            for t in range(grads.shape[1]):
                final_grads[0, t] = cv2.resize(grads[0, t], tuple(input.shape[3:]))
            return final_grads[0], output['Softmax'].data.cpu().numpy()
                
        # return gradient maps, prediction result
        return grads[0], output['Softmax'].data.cpu().numpy()
        
class GradCAM(object):
    
    def __init__(self, model, target_conv):
        self.model = model
        self.model.output_endpoints = ['Linear']
        self.model.output_endpoints.append(target_conv)
        
        self.model.net.zero_grad()
        
        self.target_conv = target_conv
        self.gradients = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        final_out = {}
        layer_index = self.model.endpoints.index(self.target_conv)
        
        for i, module in enumerate(self.model.net):
            # inter volume processing
            if self.model.endpoints[i] in self.model.inter_process.keys():
                for func, args in self.model.inter_process[self.model.endpoints[i]].items():
                    x = func(x, **args)
            
            x = module(x)
            if i == layer_index:
                # register backward hook to save gradient to be traversed to the specified layer
                x.register_hook(self.save_gradient)
                
            if self.model.endpoints[i] in self.model.output_endpoints:
                final_out[self.model.endpoints[i]] = x
            
        return final_out
        
    def generate_cam(self, input, target_class):
        # getting targeted layer output and softmax scores
        outputs = self.forward(input)
        
        # zero out non-target class output
        outputs['Linear'] = outputs['Linear'][:, target_class]
        
        # backpass from target softmax class
        outputs['Linear'].backward(retain_graph = True)
        
        # hooked gradient (filter, t, h, w)
        grad = self.gradients.data.cpu().numpy()[0]
                
        # convolution output (filter, t, h, w)
        target = outputs[self.target_conv].data.cpu().numpy()[0]
        
        # get weights from gradient, by averaging gradients of each activation (filter, t)
        weights = np.mean(grad, axis=(2, 3))
        
        # initialize empty cam array (t, h, w)
        cam = np.ones(target.shape[1:], dtype=np.float32)
        
        # multiply weights with its respective activation, sum
        for f in range(weights.shape[0]):
            for t, w in enumerate(weights[f]):
                cam[t] += w * target[f, t] # multiply grad weights with activations
                
        # cam relu
        cam = np.maximum(cam, 0)
        
        # normalization
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        
        # scale to visualizale range
        cam = np.uint8(cam * 255)
        
        # reshaping to input image size
        final_cam = np.zeros((cam.shape[0], ) + tuple(input.shape[3:]))
        for t in range(cam.shape[0]):
            final_cam[t] = cv2.resize(cam[t], tuple(input.shape[3:]), interpolation=cv2.INTER_CUBIC)
            
        final_cam = (final_cam - np.min(final_cam)) / (np.max(final_cam) - np.min(final_cam))
        
        return final_cam[0]
    
    def get_cam_img(self, img, cam):
        cam_img = np.uint8(np.zeros(img.shape))
        cam = np.uint8(cam * 255)
        
        cam_hmap = cv2.applyColorMap(cam, cv2.COLORMAP_HSV)
        
        for t in range(img.shape[0]):
            cam_img[t] = np.copy(cv2.addWeighted(img[t], 0.6, cam_hmap, 0.4, 0))
            
        return cam_img
    
    def get_grad_cam(self, grad, cam):
        if grad.shape[-1] == 3:
            cam = np.repeat(np.expand_dims(cam, 2), 3, axis = 2)
        grad_cam = np.multiply(grad, cam)
        grad_cam = np.uint8((grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam)) * 255)
        return grad_cam
    
class StreamVis(object):
    
    def __init__(self, args):
        
        self.args = torch.load(args['vis_model'])['args']
        self.device = torch.device(args['device'])
        net_config = BASE_CONFIG['network'][self.args['network']]
        
        print('visualize/initializing network')
        module = importlib.import_module('network.'+net_config['module'])
        self.model = getattr(module, net_config['class'])(self.device, BASE_CONFIG['dataset'][self.args['dataset']]['label_num'], 
                       BASE_CONFIG['channel'][self.args['modality']])
        
        print('visualize/loading model state')
        self.model.net.load_state_dict(torch.load(args['vis_model'].replace('training', 'state'), 
                                                  map_location=lambda storage, location: storage.cuda(int(args['device'][-1])) if 'cuda' in self.args['device'] else storage)['model'])
        torch.cuda.empty_cache()
        
        self.dataset_path = BASE_CONFIG['dataset'][self.args['dataset']]['base_path']
        
    def fetch_input(self):
        pass
        
if __name__ == '__main__':
    from network.r2p1d import R2P1D34Net
    device = torch.device('cuda:0')
    
    from dataloader import Videoset, transform_buffer, denormalize_buffer
    dataset = Videoset({'dataset':'UCF-101', 'modality':'rgb', 'split':1, 'is_debug_mode':0, 'debug_mode':'distributed', 
                     'debug_train_size':4, 'clip_len':8, 'resize_h':128, 'resize_w':171, 'crop_h':112, 
                     'crop_w':112, 'is_mean_sub':False, 'is_rand_flip':False, 'debug_test_size':4, 
                     'test_method':'none', 'debug_test_size':4}, 'test')
    dataset2 = Videoset({'dataset':'UCF-101', 'modality':'flow', 'split':1, 'is_debug_mode':0, 'debug_mode':'distributed', 
                     'debug_train_size':4, 'clip_len':8, 'resize_h':128, 'resize_w':171, 'crop_h':112, 
                     'crop_w':112, 'is_mean_sub':True, 'is_rand_flip':False, 'debug_test_size':4, 
                     'test_method':'none', 'debug_test_size':4}, 'test')
    input, _ = dataset2.__getitem__(0)
    #input, _ = dataset.__getitem__(len(dataset) - 2)
    input = input.requires_grad_().to(device)
    
    rgb, _ = dataset.__getitem__(0)
    
#    net = R2P1D34Net(device, 101, 3)
#    net.net.load_state_dict(torch.load(r'output\stream\UCF-101\split1\state\rgb_conv3_x.pth.tar', 
#                                       map_location=lambda storage, location: storage.cuda(0))['model'], strict=False)
    net = R2P1D34Net(device, 101, 2)
    net.net.load_state_dict(torch.load(r'output\stream\UCF-101\split1\state\flow_conv2.pth.tar', 
                                       map_location=lambda storage, location: storage.cuda(0))['model'], strict=False)
    
    print('GBP')
    gbp = GBP(net)
    grad, y = gbp.compute_grad(input, 0)
    torch.cuda.empty_cache()
    
    print('GCAM')
    gcam = GradCAM(net, 'Conv5_x')
    cam = gcam.generate_cam(input, 0)
    torch.cuda.empty_cache()
    
    img = transform_buffer(input.detach().cpu(), True)[0]
    img = denormalize_buffer(img)
    
    rgb = transform_buffer(rgb.cpu(), True)[0]
    rgb = denormalize_buffer(rgb)
    
    print('GCAM+IMG')
    #cam_img = gcam.get_cam_img(img, cam)
    cam_img = gcam.get_cam_img(rgb, cam)
        
    print('GCAM+GBP')
    #grad_cam = gcam.get_grad_cam(grad, cam)
    grad_camu = gcam.get_grad_cam(grad[:,:,:,0], cam)
    grad_camv = gcam.get_grad_cam(grad[:,:,:,1], cam)
    
    print('SOFTMAX', np.argsort(y, axis=1)[0, ::-1][:5])
    
    grad = np.uint8(((grad - np.min(grad)) / (np.max(grad) - np.min(grad))) * 255)
    
    for i in range(8):
##        grad[i] = cv2.cvtColor(grad[i], cv2.COLOR_RGB2BGR)
##        img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR)
##        cv2.imshow('img', img[i])
##        cv2.imshow('grad', grad[i])
##        cv2.imshow('cam', cam_img[i])
##        cv2.imshow('gradcam', grad_cam[i])
#        cv2.imshow('img1', img[i, :, :, 0])
#        cv2.imshow('img2', img[i, :, :, 1])
        cv2.imshow('grad1', grad[i, :, :, 0])
        cv2.imshow('grad2', grad[i, :, :, 1])
#        cv2.imshow('cam', cam_img[i])
        cv2.imshow('gradcam1', grad_camu[i])
        cv2.imshow('gradcam2', grad_camv[i])
#        
        cv2.waitKey()
        
    cv2.destroyAllWindows()