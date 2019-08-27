# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:46:46 2019

@author: Juen
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from math import ceil

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
#%%        
class GradCAM(object):
    
    def __init__(self, model, target_conv):
        self.model = model
        self.model.output_endpoints = ['Linear']
        self.model.output_endpoints.extend(target_conv)
        if self.model.convlayer[-1] not in target_conv:
            self.model.output_endpoints.append(self.model.convlayer[-1])
        
        self.model.net.zero_grad()
        
        self.target_conv = target_conv
        self.gradients = []
        
    def save_gradient(self, grad):
        self.gradients.append(grad)
        
    def forward(self, x):
        final_out = {}
        layer_index = [self.model.endpoints.index(x) for x in self.target_conv]
        
        for i, module in enumerate(self.model.net):
            # inter volume processing
            if self.model.endpoints[i] in self.model.inter_process.keys():
                for func, args in self.model.inter_process[self.model.endpoints[i]].items():
                    x = func(x, **args)
            
            x = module(x)
            if i in layer_index:
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
        grad = [x.data.cpu().numpy()[0] for x in self.gradients]
        grad.reverse()
                
        # convolution output (filter, t, h, w)
        target = [outputs[x].data.cpu().numpy()[0] for x in self.target_conv]
        
        # get weights from gradient, by averaging gradients of each activation (filter, t)
        weights = [np.mean(x, axis=(2, 3)) for x in grad]
        
        # initialize empty cam array (t, h, w)
        cam = [np.ones(x.shape[1:], dtype=np.float32) for x in target]
        
        # multiply weights with its respective activation, sum
        for ly in range(len(cam)):
            for f in range(weights[ly].shape[0]):
                for t, w in enumerate(weights[ly][f]):
                    cam[ly][t] += w * target[ly][f, t] # multiply grad weights with activations
                
        # cam relu
        cam = [np.maximum(x, 0) for x in cam]
        
        # normalization
        cam = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in cam]
        
        # scale to visualizale range
        cam = [np.uint8(x * 255) for x in cam]
        
        # reshaping to input image size
        final_cam = [np.zeros((x.shape[0], ) + tuple(input.shape[3:])) for x in cam]
        for ly in range(len(cam)):
            for t in range(cam[ly].shape[0]):
                final_cam[ly][t] = cv2.resize(cam[ly][t], tuple(input.shape[3:]), interpolation=cv2.INTER_CUBIC)
            
        final_cam = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in final_cam]
        
        return final_cam
    
    def get_cam_img(self, img, cam):
        cam_img = np.uint8(np.zeros(img.shape))
        cam = np.uint8(cam * 255)
        
        cam_hmap = [cv2.applyColorMap(x, cv2.COLORMAP_HSV) for x in cam]
        divider = img.shape[0] // len(cam_hmap)
        
        for t in range(img.shape[0]):
            cam_img[t] = np.copy(cv2.addWeighted(img[t], 0.6, cam_hmap[
                    len(cam_hmap) - 1 if (t // divider) >= len(cam_hmap) else t // divider
                    ], 0.4, 0))
            
        return cam_img
    
    def get_grad_cam(self, grad, cam):
        if grad.shape[-1] == 3:
            cam = np.repeat(np.expand_dims(cam, 3), 3, axis = 3)
            
        divider = grad.shape[0] // len(cam)
        
        grad_cam = np.zeros(grad.shape)
        c = 0
        for c in range(len(cam) - 1):
            grad_cam[c * divider : (c + 1) * divider] = np.multiply(grad[c * divider : (c + 1) * divider], cam[c])
        grad_cam[c * divider :] = np.multiply(grad[c * divider :], cam[c])
        grad_cam = np.uint8((grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam)) * 255)
        return grad_cam
    
#%%
def _cv2_frames(rgb, flow):
    rgb = transform_buffer(rgb, True)[0]
    flow = transform_buffer(flow, True)[0]
    
    rgb = denormalize_buffer(rgb)
    flow = denormalize_buffer(flow)
    flow_u = flow[:,:,:,0]
    flow_v = flow[:,:,:,1]
    frames = [rgb, flow_u, flow_v]
    
    frame_count = rgb.shape[0]
    
    img_h = 100
    img_w = 100
    img_pad = 10
    text_h = 20
    
    w_count = 8
    h_count = int(ceil(frame_count / w_count))
    
    item_h = img_h * h_count + img_pad * h_count + text_h
    item_w = img_w * w_count + img_pad * (w_count + 1)
    
    out_h = item_h * 3 + img_pad
    out_w = item_w
    
    output = np.ones((out_h, out_w, 3), np.uint8) * 255
    
    for hi, h in enumerate(['RGB', 'FLOW_U', 'FLOW_V']):
        item = np.ones((item_h, item_w, 3), np.uint8) * 255
        cv2.putText(item, h, (img_pad, img_pad * 2), 
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,0))
        for frame in range(frame_count):
            f = frame % w_count
            fh = frame // w_count
            img = cv2.resize(frames[hi][frame], (img_h, img_w))
            if hi == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img[:, :, np.newaxis]
            np.copyto(item[
                    (img_pad + img_h) * fh + img_pad + text_h : (img_pad + img_h) * (fh + 1) + text_h,
                    (img_pad + img_w) * f + img_pad : (img_pad + img_w) * (f + 1)
                    ], img)
        np.copyto(output[
                item_h * hi : item_h * (hi + 1),
                0 : item_w
                ], item)
        
#%%
def _cv2_frames_anim(rgb, flow):
    
    rgb = denormalize_buffer(rgb)[0]
    flow = denormalize_buffer(flow)[0]
    flow_u = flow[:,:,:,0]
    flow_v = flow[:,:,:,1]
    frames = [rgb, flow_u, flow_v]

    frame_count = rgb.shape[0]
    
    img_h = 120
    img_w = 120
    img_pad = 20
    text_h = 20
    
    out_h = img_h + img_pad * 2 + text_h
    out_w = (img_w + img_pad) * 3 + img_pad
    
    for t in range(frame_count):
        output = np.ones((out_h, out_w, 3), np.uint8) * 255
        cv2.putText(output, 'frame ' + str(t+1), (out_w - 80, out_h - 10), 
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255))
        
        for hi, h in enumerate(['RGB', 'FLOW_U', 'FLOW_V']):
            cv2.putText(output, h, (img_pad + (img_w + img_pad) * hi, img_pad + 10), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
            img = cv2.resize(frames[hi][t], (img_h, img_w))
            if hi == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img[:, :, np.newaxis]
            np.copyto(output[
                    img_pad + text_h : img_pad + img_h + text_h,
                    (img_pad + img_w) * hi + img_pad : (img_pad + img_w) * (hi + 1)
                    ], img)
        #cv2.imshow('o', output)
        #cv2.waitKey()
        cv2.imwrite('static/vis/preview/'+str(t)+'.jpg', output)
    #cv2.destroyAllWindows()
#%%
colors = [(109, 255, 140), (119, 119, 255)]

def _cv2_visualize_softmax(rgb, flowu, flowv, label_list, y, target_class, modality='rgb'):
    
    img_h = 140
    img_w = 140
    pad = 10
    
    bar_h = 20
    bar_w = 300
    
    text_h = 20
    
    if modality == 'rgb':
        out_h = text_h + img_h + bar_h + pad
        out_w = img_w + pad * 3 + bar_w
    else:
        out_h = text_h + img_h + bar_h + pad
        out_w = img_w * 3 + pad * 5 + bar_w
    
    output = np.ones((out_h, out_w, 3), np.uint8) * 255
    
    np.copyto(output[
            text_h + pad : text_h + pad + img_h, 
            pad : pad + img_w, :
            ], cv2.resize(rgb, (img_h, img_w)))
    if modality == 'flow':
        np.copyto(output[
                text_h + pad : text_h + pad + img_h, 
                pad * 2 + img_w : (pad + img_w) * 2, :
                ], cv2.resize(flowu, (img_h, img_w))[:, :, np.newaxis])
        np.copyto(output[
                text_h + pad : text_h + pad + img_h, 
                (pad + img_w) * 2 + pad : (pad + img_w) * 3, :
                ], cv2.resize(flowv, (img_h, img_w))[:, :, np.newaxis])
    
    top_label = np.argsort(y, axis = 1)[0, ::-1][:5]
    top_score = y[0][top_label]
    add_w = 0 if modality == 'rgb' else (img_w + pad) * 2
    
    for i, lab in enumerate(top_label):
        cv2.rectangle(output, 
                      (pad * 2 + img_w + add_w, text_h + (pad + bar_h) * i + pad), 
                      (pad * 2 + img_w + add_w + int(bar_w * top_score[i]), text_h + (pad + bar_h) * (i + 1)), 
                      colors[0 if lab == target_class else 1], cv2.FILLED)
        cv2.putText(output, label_list[lab], 
                    (pad * 2 + img_w + add_w, 
                     text_h + (pad + bar_h) * i + pad * 3), 
                     cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv2.putText(output, str(round(top_score[i] * 100, 2)), 
                    (out_w - 70, 
                     text_h + (pad + bar_h) * i + pad * 3),
                     cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        
#    cv2.imshow('out', output)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    return output
    
#%%

def _cv2_visualize_anim(rgb, flowu, flowv, label_list, layer_list, y, target_class, grads, cam_imgs, grad_cams):

    img_h = 140
    img_w = 140
    text_h = 20
    pad = 10
    
    out_h = text_h * 3 + img_h * (2 + len(cam_imgs) - 1) + pad * (2 + len(cam_imgs))
    
    modality = 'rgb' if len(flowu) == 0 and len(flowv) == 0 else 'flow'
    
    grads = np.uint8(((grads - np.min(grads)) / (np.max(grads) - np.min(grads))) * 255)
    
    for t in range(rgb.shape[0]):
        
        if modality == 'flow':
            softmax_img = _cv2_visualize_softmax(rgb[t], flowu[t], flowv[t], label_list, 
                                                 y, target_class, modality=modality)
        else:
            softmax_img = _cv2_visualize_softmax(rgb[t], [], [], label_list, 
                                                 y, target_class, modality=modality)
        
        output = np.ones((out_h, softmax_img.shape[1], 3), np.uint8) * 255
        
        # softmax img
        np.copyto(output[:softmax_img.shape[0], :, :], softmax_img)
        
        cv2.putText(output, 'RGB', (pad, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        if modality == 'flow':
            cv2.putText(output, 'FLOW_U', (pad * 2 + img_w, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
            cv2.putText(output, 'FLOW_V', ((pad + img_w) * 2 + pad, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
            cv2.putText(output, 'SOFTMAX', ((pad + img_w) * 3 + pad, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        else:
            cv2.putText(output, 'SOFTMAX', (pad * 2 + img_w, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        
        cvtCode = {'rgb' : [cv2.COLOR_RGB2BGR, cv2.COLOR_RGB2BGR, cv2.COLOR_RGB2BGR],
                   'flow' : [cv2.COLOR_GRAY2BGR, cv2.COLOR_RGB2BGR, cv2.COLOR_GRAY2BGR]
                   }
        
        # gbp
        cv2.putText(output, 'GBP', (pad, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        np.copyto(output[
                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
                pad : pad + img_w, :
                ], cv2.cvtColor(cv2.resize(grads[t], (img_h, img_w)), cvtCode[modality][0]))
        # cam (last layer)
        cv2.putText(output, 'GCAM', (pad * 2 + img_w, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        np.copyto(output[
                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
                pad * 2 + img_w : (pad + img_w) * 2, :
                ], cv2.resize(cam_imgs[-1][t], (img_h, img_w)))
        # grad cam (last layer)
        cv2.putText(output, 'GBP+GCAM', (pad * 3 + img_w * 2, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        np.copyto(output[
                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
                (pad + img_w) * 2 + pad : (pad + img_w) * 3, :
                ], cv2.cvtColor(cv2.resize(grad_cams[-1][t], (img_h, img_w)), cvtCode[modality][2]))
        
        # grad cam at other layers
        for c in range(len(cam_imgs) - 1):
            
            cv2.putText(output, layer_list[c], (pad, (img_h + pad) * c + pad + 70 + img_h + softmax_img.shape[0] + text_h), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
            
            np.copyto(output[
                softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c : softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c + img_h, 
                pad * 2 + img_w : (pad + img_w) * 2, :
                ], cv2.resize(cam_imgs[c][t], (img_h, img_w)))
    
            np.copyto(output[
                softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c : softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c + img_h, 
                (pad + img_w) * 2 + pad : (pad + img_w) * 3, :
                ], cv2.cvtColor(cv2.resize(grad_cams[c][t], (img_h, img_w)), cvtCode[modality][2]))
        
        cv2.putText(output, 'frame ' + str(t+1), (output.shape[1] - 80, 20), 
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255))
        
        cv2.imwrite('static/vis/result/'+str(t)+'.jpg', output)
    
#        cv2.imshow('out', output)
#        cv2.waitKey()
#        
#    cv2.destroyAllWindows()

#%%
    
def _cv2_visualize_output(rgb, flowu, flowv, label_list, layer_list, y, target_class, grads, cam_imgs, grad_cams, clip_name):
    
    img_h = 140
    img_w = 140
    text_h = 20
    pad = 10
    
    frame_count = rgb.shape[0]
    w_count = 8
    h_count = int(ceil(frame_count / w_count))
    
    item_h = (img_h + pad) * h_count + text_h
    item_w = (img_w + pad) * w_count + pad
    
    modality = 'rgb' if len(flowu) == 0 and len(flowv) == 0 else 'flow'
    
    if modality == 'flow':
        softmax_img = _cv2_visualize_softmax(rgb[0], flowu[0], flowv[0], label_list, y, target_class, modality=modality)
    else:
        softmax_img = _cv2_visualize_softmax(rgb[0], [], [], label_list, y, target_class, modality=modality)
    
    out_h = softmax_img.shape[0] + item_h * (2 + len(cam_imgs) * 2)
    out_w = item_w
    out_w = out_w if out_w > softmax_img.shape[1] else softmax_img.shape[1]
    
    output = np.ones((out_h, out_w, 3), np.uint8) * 255
    
    grads = np.uint8(((grads - np.min(grads)) / (np.max(grads) - np.min(grads))) * 255)
    
    np.copyto(output[:softmax_img.shape[0], :softmax_img.shape[1], :], softmax_img)
    
    title = ['GBP']
    vol = [grads]
    for i, t in enumerate(['GCAM', 'GBP+GCAM']):
        title.extend([t + ' at ' + lay for lay in layer_list])
    vol.extend([cam_imgs[i] for i in range(len(cam_imgs))])
    vol.extend([grad_cams[i] for i in range(len(grad_cams))])
    
#    cvtCode = {'rgb' : [cv2.COLOR_RGB2BGR, cv2.COLOR_RGB2BGR, cv2.COLOR_RGB2BGR],
#               'flow' : [cv2.COLOR_GRAY2BGR, cv2.COLOR_RGB2BGR, cv2.COLOR_GRAY2BGR]
#               }
    
    for i, t in enumerate(title):
        
        cv2.putText(output, t, (pad, softmax_img.shape[0] + (item_h + pad) * i), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        
        for tt in range(len(vol[i])):
            fx = tt % w_count
            fy = tt // w_count
            buf = cv2.resize(vol[i][tt], (img_h, img_w))
            if len(buf.shape) != 3:
                buf = buf[:, :, np.newaxis]
            np.copyto(output[
                    softmax_img.shape[0] + (img_h + pad) * fy + pad + (item_h + pad) * i : 
                        softmax_img.shape[0] + (img_h + pad) * (fy + 1) + (item_h + pad) * i, 
                    (pad + img_w) * fx + pad : (pad + img_w) * (fx + 1), :
                    ], buf if 'GCAM' in t or modality == 'rgb' else cv2.cvtColor(buf, cv2.COLOR_GRAY2BGR))
    
    cv2.putText(output, clip_name, (pad, pad * 2), 
                    cv2.FONT_HERSHEY_PLAIN, 1.4, (0,0,255))
    
    cv2.imwrite('output/vis/'+clip_name+'.jpg', output)
    
#%%
    
#def _cv2_visualize_flow_anim(rgb, flowu, flowv, label_list, layer_list, y, target_class, grads, cam_imgs, grad_camu, grad_camv):
#
#    img_h = 140
#    img_w = 140
#    text_h = 20
#    pad = 10
#    
#    out_h = text_h * 3 + img_h * (2 + len(cam_imgs) - 1) + pad * (2 + len(cam_imgs))
#    
#    grads = np.uint8(((grads - np.min(grads)) / (np.max(grads) - np.min(grads))) * 255)
#    
#    for t in range(rgb.shape[0]):
#        
#        softmax_img = _cv2_visualize_softmax(rgb[t], flowu[t], flowv[t], label_list, 
#                                             y, target_class, modality='flow')
#        
#        output = np.ones((out_h, softmax_img.shape[1], 3), np.uint8) * 255
#        
#        # softmax img
#        np.copyto(output[:softmax_img.shape[0], :, :], softmax_img)
#        
#        cv2.putText(output, 'RGB', (pad, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        cv2.putText(output, 'FLOW_U', (pad * 2 + img_w, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        cv2.putText(output, 'FLOW_V', ((pad + img_w) * 2 + pad, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        cv2.putText(output, 'SOFTMAX', ((pad + img_w) * 3 + pad, pad * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        
#        # gbp u
#        cv2.putText(output, 'GBP_U', (pad, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        np.copyto(output[
#                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
#                pad : pad + img_w, :
#                ], cv2.cvtColor(cv2.resize(grads[t, :, :, 0], (img_h, img_w)), cv2.COLOR_GRAY2BGR))
#        # gbp v
#        cv2.putText(output, 'GBP_V', (pad * 2 + img_w, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        np.copyto(output[
#                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
#                pad * 2 + img_w : (pad + img_w) * 2, :
#                ], cv2.cvtColor(cv2.resize(grads[t, :, :, 1], (img_h, img_w)), cv2.COLOR_GRAY2BGR))
#        # cam (last layer)
#        cv2.putText(output, 'GCAM', (pad * 3 + img_w * 2, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        np.copyto(output[
#                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
#                (pad + img_w) * 2 + pad : (pad + img_w) * 3, :
#                ], cv2.cvtColor(cv2.resize(cam_imgs[-1][t], (img_h, img_w)), cv2.COLOR_RGB2BGR))
#        # grad cam (last layer)
#        cv2.putText(output, 'GBP+GCAM_U', (pad * 4 + img_w * 3, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        np.copyto(output[
#                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
#                (pad + img_w) * 3 + pad : (pad + img_w) * 4, :
#                ], cv2.cvtColor(cv2.resize(grad_camu[-1][t], (img_h, img_w)), cv2.COLOR_GRAY2BGR))
#        # grad cam (last layer)
#        cv2.putText(output, 'GBP+GCAM_V', (pad * 5 + img_w * 4, img_h + text_h * 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        np.copyto(output[
#                softmax_img.shape[0] + text_h : softmax_img.shape[0] + text_h + img_h, 
#                (pad + img_w) * 4 + pad : (pad + img_w) * 5, :
#                ], cv2.cvtColor(cv2.resize(grad_camv[-1][t], (img_h, img_w)), cv2.COLOR_GRAY2BGR))
#    
#        # grad cam at other layers
#        for c in range(len(cam_imgs) - 1):
#            
#            cv2.putText(output, layer_list[c], (pad * 2 + img_w, (img_h + pad) * c + pad + 70 + img_h + softmax_img.shape[0] + text_h), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
#            
#            np.copyto(output[
#                softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c : softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c + img_h, 
#                (pad + img_w) * 2 + pad : (pad + img_w) * 3, :
#                ], cv2.cvtColor(cv2.resize(cam_imgs[c][t], (img_h, img_w)), cv2.COLOR_RGB2BGR))
#    
#            np.copyto(output[
#                softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c : softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c + img_h, 
#                (pad + img_w) * 3 + pad : (pad + img_w) * 4, :
#                ], cv2.cvtColor(cv2.resize(grad_camu[c][t], (img_h, img_w)), cv2.COLOR_GRAY2BGR))
#    
#            np.copyto(output[
#                softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c : softmax_img.shape[0] + text_h * 2 + img_h + (img_h + pad) * c + img_h, 
#                (pad + img_w) * 4 + pad : (pad + img_w) * 5, :
#                ], cv2.cvtColor(cv2.resize(grad_camv[c][t], (img_h, img_w)), cv2.COLOR_GRAY2BGR))
#        
#        cv2.putText(output, 'frame ' + str(t+1), (output.shape[1] - 80, 20), 
#                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255))
#        
#        cv2.imwrite('static/vis/result/'+str(t)+'.jpg', output)
#    
#        cv2.imshow('out', output)
#        cv2.waitKey()
#        
#    cv2.destroyAllWindows()
#%%
        
#def _cv2_visualize_flow(rgb, flowu, flowv, label_list, layer_list, y, target_class, grads, cam_imgs, grad_cams, clip_name):
#    
#    img_h = 140
#    img_w = 140
#    text_h = 20
#    pad = 10
#    
#    frame_count = rgb.shape[0]
#    w_count = 8
#    h_count = int(ceil(frame_count / w_count))
#    
#    item_h = (img_h + pad) * h_count + text_h
#    item_w = (img_w + pad) * w_count + pad
#    
#    softmax_img = _cv2_visualize_softmax(rgb[0], flowu[0], flowv[0], label_list, y, target_class, modality='flow')
#    
#    out_h = softmax_img.shape[0] + item_h * (2 + len(cam_imgs) * 2)
#    out_w = item_w
#    out_w = out_w if out_w > softmax_img.shape[1] else softmax_img.shape[1]
#    
#    output = np.ones((out_h, out_w, 3), np.uint8) * 255
#    
#    grads = np.uint8(((grads - np.min(grads)) / (np.max(grads) - np.min(grads))) * 255)
#    
#    # softmax img
#    np.copyto(output[:softmax_img.shape[0], :softmax_img.shape[1], :], softmax_img)
#    
#    title = ['GBP', 'GBP']
#    vol = [grads]
#    for i, t in enumerate(['GCAM', 'GBP+GCAM']):
#        title.extend([t + ' at ' + lay for lay in layer_list])
#    vol.extend([cam_imgs[i] for i in range(len(cam_imgs))])
#    vol.extend([grad_cams[i] for i in range(len(grad_cams[i]))])
#        
#    for i, t in enumerate(title):
#        
#        cv2.putText(output, t, (pad, softmax_img.shape[0] + (item_h + pad) * i), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
#        
#        for tt in range(len(vol[i])):
#            fx = tt % w_count
#            fy = tt // w_count
#            buf = cv2.resize(vol[i][tt], (img_h, img_w))
#            if len(buf.shape) != 3:
#                buf = buf[:, :, np.newaxis]
#            np.copyto(output[
#                    softmax_img.shape[0] + (img_h + pad) * fy + pad + (item_h + pad) * i : 
#                        softmax_img.shape[0] + (img_h + pad) * (fy + 1) + (item_h + pad) * i, 
#                    (pad + img_w) * fx + pad : (pad + img_w) * (fx + 1), :
#                    ], buf)
#    
#    cv2.putText(output, clip_name, (pad, pad * 2), 
#                    cv2.FONT_HERSHEY_PLAIN, 1.4, (0,0,255))
#    
#    cv2.imwrite('output/vis/'+clip_name+'.jpg', output)
#    
#class StreamVis(object):
#    def __init__(self):
#        pass
        
#%%
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
                                                  map_location=lambda storage, location: storage.cuda(int(args['device'][-1])) if 'cuda' in args['device'] else storage)['model'])
        torch.cuda.empty_cache()
        
        self.dataset_path = BASE_CONFIG['dataset'][self.args['dataset']]['base_path']
        
        self.is_ready = False
        
    def register_args(self, args, label_list):
        self.form_args = args
        self.label_list = [x[1] for x in label_list]
        
    def fetch_input(self):
        self.rgb, self.flow = read_frames_for_visualization(self.dataset_path, self.form_args['clip_name'], self.form_args['clip_len'],
                                                   self.args['crop_w'], self.args['crop_h'], self.args['is_mean_sub'])
        _cv2_frames_anim(self.rgb, self.flow)
        self.rgb = transform_buffer(self.rgb, False).to(self.device)
        self.flow = transform_buffer(self.flow, False).to(self.device)
        
        self.is_ready = True
        
    def run_visualize_rgb(self):
        # run gbp while getting softmax score
        print('visualize_rgb/running GBP')
        gbp = GBP(self.model)
        self.rgb = self.rgb.requires_grad_()
        # grad -> (t, h, w, c), y -> (1, nc)
        grad, y = gbp.compute_grad(self.rgb, int(self.form_args['target_class']))
        torch.cuda.empty_cache()
        
        print('visualize_rgb/running GCAM')
        gcam = GradCAM(self.model, self.form_args['vis_layer'])
        self.rgb = self.rgb.requires_grad_(False)
        # cam -> [layer](t/Nlayer, h, w, c)
        cam = gcam.generate_cam(self.rgb, int(self.form_args['target_class']))
        torch.cuda.empty_cache()
        
        img = transform_buffer(self.rgb.cpu(), True)[0]
        img = denormalize_buffer(img)
        for i in range(len(img)):
            img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR)
        
        # cam_img -> [layer](t, h, w, c)
        print('visualize_rgb/runnning CAM+IMG')
        cam_img = [gcam.get_cam_img(img, x) for x in cam]
        
        # grad_cam -> [layer](t, h, w, c)
        print('visualize_rgb/runnning GBP+GCAM')
        grad_cam = [gcam.get_grad_cam(grad, x) for x in cam]
        
        torch.cuda.empty_cache()
        
        # save output
        _cv2_visualize_anim(img, [], [], self.label_list, self.form_args['vis_layer'], y, 
                                int(self.form_args['target_class']), grad, cam_img, grad_cam)
        _cv2_visualize_output(img, [], [], self.label_list, self.form_args['vis_layer'], y, 
                                int(self.form_args['target_class']), grad, cam_img, grad_cam, 
                                self.args['output_name'] + '-' + self.form_args['clip_name'] + '-' + self.label_list[int(self.form_args['target_class'])])
        
    def run_visualize_flow(self):
        # run gbp while getting softmax score
        print('visualize_flow/running GBP')
        gbp = GBP(self.model)
        self.flow = self.flow.requires_grad_()
        # grad -> (t, h, w, c), y -> (1, nc) where c = 2
        grad, y = gbp.compute_grad(self.flow, int(self.form_args['target_class']))
        torch.cuda.empty_cache()
        
        # merging flow of two channels
        #grad = np.uint8(((grad - np.min(grad)) / (np.max(grad) - np.min(grad))) * 255)
        grad = np.sum(grad, axis = 3)
        
        print('visualize_flow/running GCAM')
        gcam = GradCAM(self.model, self.form_args['vis_layer'])
        self.flow = self.flow.requires_grad_(False)
        # cam -> [layer](t/Nlayer, h, w, c)
        cam = gcam.generate_cam(self.flow, int(self.form_args['target_class']))
        
        img = transform_buffer(self.rgb.cpu(), True)[0]
        img = denormalize_buffer(img)
        for i in range(len(img)):
            img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR)
        flow = transform_buffer(self.flow.cpu(), True)[0]
        flow = denormalize_buffer(flow)
        flowu = flow[:, :, :, 0]
        flowv = flow[:, :, :, 1]
        torch.cuda.empty_cache()
        
        # cam_img -> [layer](t, h, w, c)
        print('visualize_flow/runnning CAM+IMG')
        cam_img = [gcam.get_cam_img(img, x) for x in cam]
        
        # grad_cam -> [layer](t, h, w, c)
        print('visualize_flow/runnning GBP+GCAM')
        grad_cam = [gcam.get_grad_cam(grad, x) for x in cam]
        #grad_cam_u = [gcam.get_grad_cam(grad[:, :, :, 0], x) for x in cam]
        #grad_cam_v = [gcam.get_grad_cam(grad[:, :, :, 1], x) for x in cam]
        
        torch.cuda.empty_cache()
        
        # save output
        _cv2_visualize_anim(img, flowu, flowv, self.label_list, self.form_args['vis_layer'], y, 
                            int(self.form_args['target_class']), grad, cam_img, grad_cam)
        _cv2_visualize_output(img, flowu, flowv, self.label_list, self.form_args['vis_layer'], y, 
                              int(self.form_args['target_class']), grad, cam_img, grad_cam, 
                                self.args['output_name'] + '-' + self.form_args['clip_name'] + '-' + self.label_list[int(self.form_args['target_class'])])
        
#%%
        
if __name__ == '__main__':
    from network.r2p1d import R2P1D34Net
    device = torch.device('cuda:0')
    
    rgb, flow = read_frames_for_visualization('C:\\Users\\Juen\\Desktop\\Gabumon\\Blackhole\\UTAR\\Subjects\\FYP\\dataset\\UCF-101', 
                                  'v_ApplyEyeMakeup_g08_c01', 8, 112, 112)
    
    rgb = transform_buffer(rgb, False).to(device).requires_grad_()
    flow = transform_buffer(flow, False).to(device).requires_grad_()
    
    net = R2P1D34Net(device, 101, 3)
    net.net.load_state_dict(torch.load(r'output\stream\UCF-101\split1\state\rgb_conv3_x.pth.tar', 
                                       map_location=lambda storage, location: storage.cuda(0))['model'], strict=False)
#    net = R2P1D34Net(device, 101, 2)
#    net.net.load_state_dict(torch.load(r'output\stream\UCF-101\split1\state\flow_conv2.pth.tar', 
#                                       map_location=lambda storage, location: storage.cuda(0))['model'], strict=False)
    
    f_in = open(r'C:\Users\Juen\Desktop\Gabumon\Blackhole\UTAR\Subjects\FYP\dataset\UCF-101\classInd.txt')
    label_list = [x.split(' ')[1] for x in (f_in.read().split('\n'))[:-1]]
    f_in.close()
    
    print('GBP')
    gbp = GBP(net)
    grad, y = gbp.compute_grad(rgb, 0)
    gbp.de_init()
    torch.cuda.empty_cache()
    
    print('GCAM')
    flow = flow.requires_grad_(False)
    rgb = rgb.requires_grad_(False)
    layer_list = ['Conv2_x', 'Conv3_x', 'Conv5_x']
    gcam = GradCAM(net, ['Conv2_x', 'Conv3_x', 'Conv5_x'])
    cam = gcam.generate_cam(rgb, 0)
    torch.cuda.empty_cache()
    
#    img = transform_buffer(input.detach().cpu(), True)[0]
#    img = denormalize_buffer(img)
    
    rgb = transform_buffer(rgb.cpu(), True)[0]
    rgb = denormalize_buffer(rgb)
    
#    flow = transform_buffer(flow.cpu(), True)[0]
#    flow = denormalize_buffer(flow)
#    flowu = flow[:, :, :, 0]
#    flowv = flow[:, :, :, 1]
    
    print('GCAM+IMG')
    #cam_img = gcam.get_cam_img(img, cam)
    cam_img = [gcam.get_cam_img(rgb, x) for x in cam]
        
    print('GCAM+GBP')
    grad_cam = [gcam.get_grad_cam(grad, x) for x in cam]
#    grad_camu = [gcam.get_grad_cam(grad[:,:,:,0], x) for x in cam]
#    grad_camv = [gcam.get_grad_cam(grad[:,:,:,1], x) for x in cam]
#    
#    print('SOFTMAX', np.argsort(y, axis=1)[0, ::-1][:5])
#    
#    grad = np.uint8(((grad - np.min(grad)) / (np.max(grad) - np.min(grad))) * 255)
#    for i in range(len(cam_img)):
#        for t in range(8):
#            cv2.imshow('cam_img', cam_img[i][t])
#            cv2.imshow('grad_cam', grad_cam[i][t])
#            cv2.waitKey()
#    cv2.destroyAllWindows()
#    
#    for i in range(8):
###        grad[i] = cv2.cvtColor(grad[i], cv2.COLOR_RGB2BGR)
###        img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR)
###        cv2.imshow('img', img[i])
###        cv2.imshow('grad', grad[i])
###        cv2.imshow('cam', cam_img[i])
###        cv2.imshow('gradcam', grad_cam[i])
##        cv2.imshow('img1', img[i, :, :, 0])
##        cv2.imshow('img2', img[i, :, :, 1])
#        cv2.imshow('grad1', grad[i, :, :, 0])
#        cv2.imshow('grad2', grad[i, :, :, 1])
##        cv2.imshow('cam', cam_img[i])
#        cv2.imshow('gradcam1', grad_camu[i])
#        cv2.imshow('gradcam2', grad_camv[i])
##        
#        cv2.waitKey()
#        
#    cv2.destroyAllWindows()