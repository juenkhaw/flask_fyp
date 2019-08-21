# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:27:12 2019

@author: Juen
"""

from torch.utils.data import Dataset
from torch import Tensor

import numpy as np
import cv2
from random import random

from glob import glob
from os import listdir, path, makedirs
from shutil import copy2

from flask_fyp import BASE_CONFIG

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
#                "label_index_txt" : "classInd.txt", 
#                "train_txt" : ["hmdb_trainlist01.txt", "hmdb_trainlist02.txt", "hmdb_trainlist03.txt"],
#                "val_txt" : [], 
#                "test_txt" : ["hmdb_testlist01.txt", "hmdb_testlist02.txt", "hmdb_testlist03.txt"]
#                }
#    }
#}
                
def generate_subbatches(sbs, *tensors):
    """
    Generate list of subbtaches from a batch of sample data
    sbs: sub batch size
    tensors : list of tensors batch to be partitioned  
    Returns:
        subbatches : list of partitioned subbateches
    """
    # engulf tensor into a list if there is only one tensor passed in
    if isinstance(tensors, Tensor):
        tensors = [tensors]
        
    subbatches = []
    for i in range(len(tensors)):
        subbatch = []
        part_num = tensors[i].shape[0] // sbs
        # if subbatch size is lower than the normal batch size
        if sbs < tensors[i].shape[0]:
            # partitioning batch with subbatch size
            subbatch = [tensors[i][j * sbs : j * sbs + sbs] for j in range(part_num)]
            # if there is any remainder in the batch
            if part_num * sbs < tensors[i].shape[0]:
                subbatch.append(tensors[i][part_num * sbs : ])
            subbatches.append(subbatch)
        else:
            subbatches.append([tensors[i]])
    
    return subbatches if len(tensors) > 1 else subbatches[0]

#%%              
def transform_buffer(buffer, to_display):
    """
    Transform frame/clip/video(multiple clips) matrix to/from tensor and displayable
    buffer: frame Tensor
    to_display: conversion mode (Tensor <-> numpy ndarray)
    ===========NOTES=========
    Tensor format---------(clip_num, channel_num, length, height, width)
    Disaplayable format---(clip_num, length, height, width, channel_num)
    """
    if len(buffer.shape) == 3: # single frame
        if to_display:
            return buffer.numpy().transpose(1, 2, 0)
        else:
            return Tensor(buffer.transpose(2, 0, 1))
    
    elif len(buffer.shape) == 4: # clip (multiple frames)
        if to_display:
            return buffer.numpy().transpose(1, 2, 3, 0)
        else:
            return Tensor(buffer.transpose(3, 0, 1, 2))
        
    elif len(buffer.shape) == 5: # video (multiple clips)
        if to_display:
            return buffer.numpy().transpose(0, 2, 3, 4, 1)
        else:
            return Tensor(buffer.transpose(0, 4, 1, 2, 3))
    else: #error
        return None
    
#%%
                
def temporal_crop(buffer_len, clip_len):
    """
    Randomly extracts frames over temporal dimension
    buffer_len : original video framecount (depth)
    clip_len : expected output clip framecount
        
    Returns:
        start_index, end_index : starting and ending indices of the clip
    """
    sample_len = clip_len + clip_len / 2
    # randomly select time index for temporal jittering
    if buffer_len > sample_len:
        start_index = np.random.randint(buffer_len - clip_len)
    else:
        if buffer_len != sample_len:
            multiplier = int(np.ceil(sample_len / buffer_len))
        else:
            multiplier = 2
        start_index = np.random.randint(buffer_len * multiplier - clip_len)
    end_index = start_index + clip_len
    
    return start_index, end_index
    
def temporal_center_crop(buffer_len, clip_len):
    """
    Extracts center of temporal portion of video
    buffer_len : original video framecount (depth)
    clip_len : expected output clip framecount
    
    Returns:
        start_index, end_index : starting and ending indices of the clip
    """
    if buffer_len >= clip_len:
        start_index = (buffer_len - clip_len) // 2
    else:
        multiplier = int(np.ceil(clip_len / buffer_len))
        start_index = (buffer_len * multiplier - clip_len) // 2
    end_index = start_index + clip_len
    
    return start_index, end_index
#%%
def temporal_uniform_crop(buffer_len, clip_len, clips_per_video):
    """
    Uniformly crops the video into N clips over temporal dimension
    buffer_len : original video framecount (depth)
    clip_len : expected output clip framecount
    clips_per_video : number of clips for each video sample
        
    Returns:
        indices : list of starting and ending indices of each clips
    """
    while buffer_len < (clip_len + clip_len / 2):
        buffer_len *= 2
    
    # compute the average spacing between each consecutive clips
    # could be negative if buffer_len < clip_len * clips_per_video
    spacing = (buffer_len - clip_len * clips_per_video) / (clips_per_video - 1)
    
    indices = [(0, clip_len)]
    for i in range(1, clips_per_video - 1):
        start = round(indices[i - 1][1] + spacing)
        end = start + clip_len
        indices.append((start, end))
        
    indices.append((buffer_len - clip_len, buffer_len))
    
    return indices
#%%
def spatial_crop(buffer_size, clip_size):
    """
    Randomly crops original video frames over spatial dimension
    buffer_size : size of the original scaled frames
    clip_size : size of the output clip frames
        
    Returns:
        start_h, start_w : (x, y) point of the top-left corner of the patch
        end_h, end_w : (x, y) point of the bottom-right corner of the patch
    """
    
    # expected parameters to be a tuple of height and width
    assert(len(buffer_size) == 2 and len(clip_size) == 2)
    
    # randomly select start indices in spatial dimension to crop the video
    start_h = np.random.randint(buffer_size[0] - clip_size[0])
    end_h = start_h + clip_size[0]
    start_w = np.random.randint(buffer_size[1] - clip_size[1])
    end_w = start_w + clip_size[1]
    
    return (start_h, end_h), (start_w, end_w)

def spatial_center_crop(buffer_size, clip_size):
    """
    Crops a center patch from frames
    buffer_size : size of the original scaled frames
    clip_size : size of the output clip frames
        
    Returns:
        start_h, start_w : (x, y) point of the top-left corner of the patch
        end_h, end_w : (x, y) point of the bottom-right corner of the patch
    """
    
    # expected parameters to be a tuple of height and width
    assert(len(buffer_size) == 2 and len(clip_size) == 2)
    
    # obtain top-left and bottom right coordinate of the center patch
    start_h = (buffer_size[0] - clip_size[0]) // 2
    end_h = start_h + clip_size[0]
    start_w = (buffer_size[1] - clip_size[1]) // 2
    end_w = start_w + clip_size[1]
    
    return (start_h, end_h), (start_w, end_w)

def spatial_10_crop(buffer_size, clip_size):
    """
    buffer_size : size of the original scaled frames
    clip_size : size of the output clip frames
    """
    # expected parameters to be a tuple of height and width
    assert(len(buffer_size) == 2 and len(clip_size) == 2)
    
    h_top = (0, clip_size[0])
    h_btm = (buffer_size[0] - clip_size[0], buffer_size[0])
    w_left = (0, clip_size[1])
    w_right = (buffer_size[1] - clip_size[1], buffer_size[1])
    
    return [(h_top, w_left), (h_top, w_right), (h_btm, w_left), (h_btm, w_right), spatial_center_crop(buffer_size, clip_size)]

def spatial_random_flip(buffer):
    """
    Flips a sequence of frames spatially by a random 50% chance
    buffer: displayable tensor
    """
    if random() > 0.5:
        # np.flip undergoes some [:::-1] operation, which Torch couldnt support
        #return Tensor(np.flip(buffer.numpy(), axis = len(buffer.shape) - 2).copy())
        return np.flip(buffer, axis = len(buffer.shape) - 2)
    else:
        return buffer

def normalize_buffer(buffer):
    """
    Normalizes values of input frames buffer
    buffer : unnormalized frames data
    """
    #normalize the pixel values to be in between -1 and 1
    buffer = (buffer - 128) / 128
    return buffer

#%%
def denormalize_buffer(buffer):
    """
    Denormalizes intensity values of input buffer frames
    buffer : normalized frames data
    """
    buffer = (buffer * 128 + 128).astype(np.uint8) 
    return buffer
#%%

def flow_mean_sub(buffer):
    """
    Elimiates global flow intensity to better identify flow encoded in region-in-interest
    buffer: 5-D tensor (multiple clips)
    """
     
    # buffer shape (clip_num, depth, h, w, chnl)
    # buffer sum of shape (clip_num, h, w, chnl)
    sh = buffer.shape
    
    # sum and average over the depth (l) axis
    buffer_mean = np.sum(buffer, axis = 1) / sh[1]
    
    # substract the mean
    buffer -= np.expand_dims(buffer_mean, 1)
    # shift range of data up to eliminate negative value
    buffer += np.abs(np.min(buffer, axis = (1, 2, 3))).reshape(sh[0], 1, 1, 1, sh[4])
    # normalize
    buffer = buffer / np.max(buffer, axis = (1, 2, 3)).reshape(sh[0], 1, 1, 1, sh[4]) * 255

    return buffer

class Videoset(Dataset):
    
    def __init__(self, args, mode):
        """
        Initializes videoset by reading paths to all video files of specified split/set
        args: train_stream form arguments dict
        mode: train/val/test
        """
        
        self._dataset_info = BASE_CONFIG['dataset'][args['dataset']]
        self._args = args
        self._mode = mode
        
        # making sure the dataset path given contains rgb and flow folders
        assert(path.exists(self._dataset_info['base_path']))
        assert(all([x in listdir(self._dataset_info['base_path']) for x in ['rgb', 'flow']]))
        assert(all([x in listdir(path.join(self._dataset_info['base_path'], 'flow')) for x in ['u', 'v']]))
        
        super(Videoset, self).__init__()
        
        # concatenate dataset base dir with specified modality
        self._base_dir = path.join(self._dataset_info['base_path'], args['modality'])
        
        # read in list of class label
        f_in = open(path.join(self._dataset_info['base_path'], self._dataset_info['label_index_txt']))
        self._label_list = [x.split(' ')[1] for x in (f_in.read().split('\n'))[:-1]]
        f_in.close()
        
        # read in path to each of the sample in specified split/set
        f_in = open(path.join(self._dataset_info['base_path'], self._dataset_info[mode+'_txt'][args['split'] - 1]))
        buffer = f_in.read().split('\n')[:-1]
        self._Y = np.array([x.split(' ')[1] for x in buffer], dtype = np.int)
        if args['modality'] == 'rgb':
            self._X_path = np.array([[x.split(' ')[0]] for x in buffer], dtype=np.str)
        else:
            self._X_path = np.array([[path.join(y, x.split(' ')[0]) for y in ['u', 'v']] for x in buffer])
            
        # trim sample set if debugging mode is enabled
        if args['is_debug_mode']:
            indices = None
            if args['debug_mode'] == 'peek':
                indices = list(range(args['debug_'+mode+'_size']))
                
            else:
                sample_num = args['debug_'+mode+'_size']
                # ensure that sample number does not exceed the freq of least occurence class label
                self._Y_freq = np.bincount(self._Y)
                assert(sample_num <= np.min(self._Y_freq))
                
                # extract indices of sample to be selected
                indices = []
                count = 0
                for f in self._Y_freq:
                    indices.extend(list(range(count, count + sample_num)))
                    count += f
                    
            self._X_path = self._X_path[indices]
            self._Y = self._Y[indices]
            
        # count on occurence of each label
        self._Y_freq = np.bincount(self._Y)
        
        self._read_first_frame_forach_label()
            
    def __getitem__(self, index):
        
        # read in all content within a sample folder
        path_content = []
        for p in self._X_path[index]:
            path_content.append(glob(path.join(self._base_dir, p, '*.jpg')))
            path_content.sort() # ensure the frame order is correct
            
        # retrieve frame count of current sample
        frame_count = len(path_content[0])
        
        t_index = []
        s_index = None
        if self._mode in ['train', 'val']:
            # retrieve temporal indices
            if self._mode == 'train': # random temporal spatial cropping
                t_index.append(temporal_crop(frame_count, self._args['clip_len']))
                s_index = spatial_crop((self._args['resize_h'], self._args['resize_w']), 
                                       (self._args['crop_h'], self._args['crop_w']))
            else: # center temporal spatial cropping
                t_index.append(temporal_center_crop(frame_count, self._args['clip_len']))
                s_index = spatial_center_crop((self._args['resize_h'], self._args['resize_w']), 
                                       (self._args['crop_h'], self._args['crop_w']))
        else:
            if self._args['test_method'] == '10-clips':
                # retrieve indices for center cropping and temporal index for each clips
                t_index = temporal_uniform_crop(frame_count, self._args['clip_len'], 10)
                s_index = spatial_center_crop((self._args['resize_h'], self._args['resize_w']), 
                                              (self._args['crop_h'], self._args['crop_w']))
            elif self._args['test_method'] == '10-crops': # 10-crops
                t_index = temporal_uniform_crop(frame_count, self._args['clip_len'], 10)
                s_index = spatial_10_crop((self._args['resize_h'], self._args['resize_w']), 
                                              (self._args['crop_h'], self._args['crop_w']))
            else:
                t_index.append(temporal_center_crop(frame_count, self._args['clip_len']))
                s_index = spatial_center_crop((self._args['resize_h'], self._args['resize_w']), 
                                       (self._args['crop_h'], self._args['crop_w']))
            
        clip_count = 10 if self._mode == 'test' and self._args['test_method'] != 'none' else 1
        if clip_count == 10 and self._args['test_method'] == '10-crops':
            clip_count = 100
        channel_count = 3 if self._args['modality'] == 'rgb' else 2
        buffer = np.empty((clip_count, self._args['clip_len'], self._args['crop_h'], 
                           self._args['crop_w'], channel_count), np.float32)
        
        for t in range(len(t_index)):
            # extract only frames with indices specified
            frame = t_index[t][0]
            while frame < t_index[t][1]:
                # ensure the frame index always lower than total frame count
                frame2 = frame
                while frame2 >= frame_count: frame2 -= frame_count
                
                buffer_frame = []
                # read in rgb frames
                if channel_count == 3:
                    buffer_frame.append(cv2.imread(path_content[0][frame2]))
                # read in flow maps
                else:
                    buffer_frame.append(cv2.imread(path_content[0][frame2], cv2.IMREAD_GRAYSCALE))
                    buffer_frame.append(cv2.imread(path_content[1][frame2], cv2.IMREAD_GRAYSCALE))
                                        
                for i in range(len(buffer_frame)):
                    if buffer_frame[i] is not None:
                        
                        if self._args['test_method'] == 'none' and self._mode == 'test':
                            
                            buffer_frame[i] = cv2.resize(buffer_frame[i], (self._args['crop_w'], self._args['crop_w']))
                            
                            # copying to the buffer tensor
                            if channel_count == 3:
                                np.copyto(buffer[t, frame - t_index[t][0], :, :, :], buffer_frame[i])
                            else:
                                np.copyto(buffer[t, (frame - t_index[t][0]), :, :, i], buffer_frame[i][:, :])
                            
                        else:
                            # applying resizing
                            buffer_frame[i] = cv2.resize(buffer_frame[i], (self._args['resize_w'], self._args['resize_h']))
                            
                            # add channel dimension for flow maps
                            if channel_count == 2:
                                buffer_frame[i] = buffer_frame[i][:, :, np.newaxis]
                            
                            if self._args['test_method'] == '10-crops':
                                for s in range(len(s_index)):
                                    buffer_frame2 = buffer_frame[i][s_index[s][0][0] : s_index[s][0][1], 
                                                 s_index[s][1][0] : s_index[s][1][1], :]
                                    if channel_count == 3:
                                        np.copyto(buffer[t * 10 + s, frame - t_index[t][0], :, :, :], buffer_frame2)
                                        np.copyto(buffer[t * 10 + s + 5, frame - t_index[t][0], :, :, :], np.flip(buffer_frame2, axis = 1))
                                    else:
                                        np.copyto(buffer[t * 10 + s, (frame - t_index[t][0]), :, :, i], buffer_frame2[:, :, 0])
                                        np.copyto(buffer[t * 10 + s + 5, (frame - t_index[t][0]), :, :, i], np.flip(buffer_frame2[:, :, 0], axis = 1))
                            
                            else: # other besides 10-crop testing
                                # applying random cropping for training and center cropping for validation/10-clips testing
                                buffer_frame[i] = buffer_frame[i][s_index[0][0] : s_index[0][1], 
                                         s_index[1][0] : s_index[1][1], :]
                                
                                # copying to the buffer tensor
                                if channel_count == 3:
                                    np.copyto(buffer[t, frame - t_index[t][0], :, :, :], buffer_frame[i])
                                else:
                                    np.copyto(buffer[t, (frame - t_index[t][0]), :, :, i], buffer_frame[i][:, :, 0])
                            
                    else:
                        print(path_content[i][frame2])
                        assert(1==0)
                        
                frame += 1
                
            # mean substraction
            if self._args['is_mean_sub'] and channel_count == 2:
                buffer = flow_mean_sub(buffer)
                
            # random vertical flipping
            if self._args['is_rand_flip'] and self._mode == 'train':
                buffer = spatial_random_flip(buffer)
            
        # normalization and transform to Tensor format
        buffer = normalize_buffer(buffer)
        if self._mode in ['train', 'val']:
            return transform_buffer(buffer, False)[0], self._Y[index]
        else:
            return transform_buffer(buffer, False), self._Y[index]
        
    def __len__(self):
        return len(self._Y)
            
    def _read_first_frame_forach_label(self):
        """
        INTERNAL USE ONLY
        """
        index = 0
        save_path = path.join('static', self._args['dataset'], 'split'+str(self._args['split']), self._mode)
        if not path.exists(save_path):
            makedirs(save_path)
        else:
            if len(glob(path.join(save_path, '*.jpg'))) == self._dataset_info['label_num']: return
        for i in range(self._dataset_info['label_num']):
            if i == len(self._Y_freq):
                break
            if self._Y_freq[i] == 0:
                continue
            else:
                #print(i)
                current_sample = glob(path.join(self._base_dir, self._X_path[index][0], '*.jpg'))[0]
                copy2(current_sample, path.join(save_path,self._label_list[i]+'.jpg'))
                index += self._Y_freq[i]
#%%
def read_frames_for_visualization(dataset_path, video_name, clip_len, spat_w, spat_h, mean_sub = True):
    
    # read path
    paths = [glob(path.join(dataset_path, 'rgb', video_name, '*jpg')), 
             glob(path.join(dataset_path, 'flow', 'u', video_name, '*jpg')),
             glob(path.join(dataset_path, 'flow', 'v', video_name, '*jpg'))]
    for p in paths: p.sort()
    
    # extract middle section of video
    t_index = temporal_center_crop(len(paths[0]), clip_len)
    
    buffer_frames = []
    for p, m in enumerate([3, 1, 1]):
        buffer_frame = np.empty((1, clip_len, spat_h, spat_w, m))
        for t in range(t_index[0], t_index[1]):
            
            ft = t
            while ft >= len(paths[p]): ft -= len(paths[p])
            
            buffer = cv2.imread(paths[p][ft], cv2.IMREAD_COLOR if m == 3 else cv2.IMREAD_GRAYSCALE)
            buffer = cv2.resize(buffer, (spat_w, spat_h))
            if m == 3:
                buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2RGB)
            else:
                buffer = buffer[:,:,np.newaxis]
            np.copyto(buffer_frame[0, t - t_index[0]], buffer)
        if m == 1 and mean_sub:
            buffer_frame = flow_mean_sub(buffer_frame)
        buffer_frame = normalize_buffer(buffer_frame)
        buffer_frames.append(buffer_frame)
    
    # concat flows
    rgb_frame = buffer_frames[0]
    flow_frame = np.concatenate((buffer_frames[1], buffer_frames[2]), axis = 4)
    
    return rgb_frame, flow_frame
#%%
if __name__ == '__main__':
#    rgb, flow = read_frames_for_visualization('C:\\Users\\Juen\\Desktop\\Gabumon\\Blackhole\\UTAR\\Subjects\\FYP\\dataset\\UCF-101', 
#                                  'v_ApplyEyeMakeup_g08_c01', 8, 112, 112)
#    rgb = transform_buffer(rgb, True)
#    flow = transform_buffer(flow, True)
#    rgb = denormalize_buffer(rgb)
#    flow = denormalize_buffer(flow)
#    for i in range(8):
#        cv2.imshow('rgb', rgb[0, i])
#        cv2.imshow('flow1', flow[0, i, :, :, 0])
#        cv2.imshow('flow2', flow[0, i, :, :, 1])
#        cv2.waitKey()
#    cv2.destroyAllWindows()
    temp = Videoset({'dataset':'UCF-101', 'modality':'rgb', 'split':1, 'is_debug_mode':0, 'debug_mode':'distributed', 
                     'debug_train_size':4, 'clip_len':64, 'resize_h':128, 'resize_w':171, 'crop_h':112, 
                     'crop_w':112, 'is_mean_sub':False, 'is_rand_flip':False, 'debug_test_size':4, 
                     'test_method':'10-clips', 'debug_test_size':4}, 'test')
#    #temp._read_first_frame_forach_label()
    a = temp.__getitem__(2144)
#    at = transform_buffer(a, True)
#    for i in range(8):
#        #cv2.imshow('t', cv2.cvtColor(at[4, i], cv2.COLOR_RGB2BGR))
#        cv2.imshow('u', at[9, i, :, :, 0])
#        cv2.imshow('v', at[9, i, :, :, 1])
#        cv2.waitKey()
#    cv2.destroyAllWindows()
    #temp = Videoset({'dataset':'HMDB-51', 'modality':'rgb', 'split':1}, 'train')