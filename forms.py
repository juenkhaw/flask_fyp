# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:23:35 2019

@author: Juen
"""

from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, DecimalField, SubmitField, ValidationError, FileField, IntegerField, BooleanField, SelectMultipleField
from wtforms.validators import InputRequired, regexp

from . import BASE_CONFIG

import os
import glob

def num_range(min = None, max = None, dependant = None):
    if dependant == None:
        errors = ['Must not be less than ' + str(min), 'Must not be more than ' + str(max)]
    else:
        errors = ['Must not be less than ' + (str(min) if dependant[0] == None else dependant[0]),
                  'Must not be more than ' + (str(max) if dependant[1] == None else dependant[1])]
    
    def _num_range(form, field):
        if min != None:
            if field.data < min:
                raise ValidationError(errors[0])
        if max != None:
            if field.data > max:
                raise ValidationError(errors[1])
                
    return _num_range

def validate_dataset_path(form, field):
    try:
        dirs = os.listdir(field.data)
    except:
        raise ValidationError('The path specified could not be read')
    
    if not all(x in dirs for x in ['rgb', 'flow']) :
        raise ValidationError('The path specified does not contain rgb/flow folder')
        
def validate_empty_select(form, field):
    if field.data == '':
        raise ValidationError('This selection is empty')
        
def selection_limit(limit):
    
    def _selection_limit(form, field):
        if len(field.data) > limit:
            raise ValidationError('Choose NO more than '+str(limit))
    
    return _selection_limit

class TrainStreamForm(FlaskForm):
    
    _gpu_names = []
    _split_choices = []
    
    # resume training
    half_model = SelectField(u'Select model to start training', validators=[])
    
    # basic setup
    modality = SelectField(u'Modality', choices = [('rgb','RGB'), ('flow','FLOW')], validators=[InputRequired()])
    dataset = SelectField(u'Video Dataset', choices = [(x, str(x)) for x in list(BASE_CONFIG['dataset'].keys())], validators=[InputRequired()])
    split = SelectField(u'Split', coerce = int, choices = [], validators=[InputRequired()], id = 'split_select')
    network = SelectField(u'Network Architecture', choices = [(x, x) for x in list(BASE_CONFIG['network'].keys())], validators=[InputRequired()])
    device = SelectField(u'Device', choices = [], validators=[InputRequired()])
    epoch = IntegerField(u'Epoch', validators=[InputRequired(), num_range(min = 1)], default = 3) ###
    pretrain_model = SelectField(u'Pretrained Model', choices = [], validators=[InputRequired()])
    freeze_point = SelectField(u'Freeze Network (up to this point, exclusive)', coerce = str, choices = [], validators=[InputRequired()])
    
    # optimizer
    base_lr = DecimalField(u'Base Learning Rate', places = None, default = 0.01, validators=[InputRequired(), num_range(min = 0)])
    batch_size = IntegerField(u'Batch Size', validators=[InputRequired(), num_range(min = 1)], default = 32)
    is_batch_size = BooleanField(u'Apply the same batch size to subbatch size')
    momentum = DecimalField(u'Momentum', places = None, default = 0.1, validators=[InputRequired(), num_range(min = 0)])
    l2decay = DecimalField(u'L2 Weight Decay', places = None, default = 0.01, validators=[InputRequired(), num_range(min = 0)])
    dropout = DecimalField(u'Dropout Ratio (Prob to be excluded)', places = None, default = 0, validators=[InputRequired(), num_range(min = 0, max = 1)])
    # for subbatch size
    sub_batch_size = IntegerField(u'Training Sub-Batch Size', validators=[InputRequired(), num_range(min = 1, max = 32, dependant=[None, 'batch_size'])], default = 3) ###
    val_batch_size = IntegerField(u'Val Batch Size', validators=[InputRequired(), num_range(min = 1, max = 32, dependant=[None, 'batch_size'])], default = 8) ###
    # for optmizer scheduler
    lr_scheduler = SelectField(u'LR Schedule Scheme', choices=[('none', 'None'), ('stepLR', 'Reduce by Epoch'), ('dynamic', 'Reduce on Plateau')], validators=[InputRequired()], default = 'dynamic') ###
    lr_reduce_ratio = DecimalField(u'LR Decay Factor', places = None, default = 0.1, validators=[InputRequired()])
    # for reduce by step
    step_size = IntegerField(u'Epoch Step', default = 10, validators=[InputRequired(), num_range(min = 1)])
    last_step = IntegerField(u'Last Epoch Stopping Reduction', default = -1, validators=[InputRequired(), num_range(min = -1)])
    # for reduce on plateau
    patience = IntegerField(u'Patience Epoch Step', default = 10, validators=[InputRequired(), num_range(min = 0)])
    loss_threshold = DecimalField(u'Optimum Loss Threshold', places = None, default = 0.0001, validators=[InputRequired()])
    min_lr = DecimalField(u'Minimum LR', places = None, default = 0, validators=[InputRequired(), num_range(min = 0, max = 0.01, dependant=[None, 'base LR'])])
    
    # data preprocessing
    clip_len = IntegerField(u'Clip Length', validators=[InputRequired(), num_range(min = 1)], default = 8)
    resize_h = IntegerField(u'Resize Height', default = 128, validators=[InputRequired(), num_range(min = 1)])
    resize_w = IntegerField(u'Resize Width', default = 171, validators=[InputRequired(), num_range(min = 1)])
    crop_h = IntegerField(u'Spatial Height', default = 112, validators=[InputRequired(), num_range(min = 1)])
    crop_w = IntegerField(u'Spatial Width', default = 112, validators=[InputRequired(), num_range(min = 1)])
    is_mean_sub = BooleanField(u'Enable Flow Mean Substraction')
    is_rand_flip = BooleanField(u'Enable Random Vertical Flipping')
    
    # debugging mode
    is_debug_mode = BooleanField(u'Enable Debugging Mode', default = True) ###
    debug_mode = SelectField(u'Data Selection Mode', choices = [('peek','Peek (select first N samples regardless of label)'), ('distributed','Distributed (select first N samples across all labels)')], validators=[InputRequired()])
    debug_train_size = IntegerField(u'Train Set Size', default = 8, validators=[InputRequired()]) ###
    debug_val_size = IntegerField(u'Validation Set Size', default = 8, validators=[InputRequired()]) ###
    
    # output
    output_name = StringField(u'Output File Name')
    output_compare = SelectMultipleField(u'Comparing with Other Models')
    
    # hidden field
    test_method = StringField(u'', default='none')

    submit = SubmitField('Start')
    
    def __init__(self, gpu_names):
        super(TrainStreamForm, self).__init__()
        self._gpu_names = [('cuda:'+str(i), 'CUDA:'+str(i)+' '+gpu_names[i]) for i in range(len(gpu_names))]
        self._gpu_names.extend([('cpu', 'CPU')])
        self.device.choices = self._gpu_names
        
        first_dataset = list(BASE_CONFIG['dataset'].keys())[0]
        self.split.choices = [(x, x) for x in list(range(1, BASE_CONFIG['dataset'][first_dataset]['split'] + 1))]
        
        self.pretrain_model.choices = [('none', 'None')]
        self.pretrain_model.choices.extend([(x, x.split('.')[0]) for x in os.listdir('pretrained') if '.pth.tar' in x])
        
        #self.output_compare.choices = [(x, x) for x in os.listdir('output/stream/training') if '.pth.tar' in x]
        self.output_compare.choices = [(x,'-'.join([x.split('.')[0].split('\\')[i] for i in [2, 3, 5]]))
            for x in glob.glob(r'output\stream\**\training\*.pth.tar', recursive=True)]
        
        self.freeze_point.choices = [('none', 'None')]
        for net in BASE_CONFIG['network'].keys():
            self.freeze_point.choices.extend([(x, x) for x in BASE_CONFIG['network'][net]['endpoint']])
            
        self.half_model.choices = [('', 'Start a new one')]
        # making sure the state and output files are all both existed for the models
        self.half_model.choices.extend([(x,'-'.join([x.split('.')[0].split('\\')[i] for i in [2, 3, 5]]))
            for x in glob.glob(r'output\stream\**\training\*.pth.tar', recursive=True)])
            
class ResumeStreamForm(FlaskForm):
    
    # basic setup
    half_model = SelectField(u'Select Model', validators=[InputRequired(), validate_empty_select])
    device = SelectField(u'Device', choices = [], validators=[InputRequired()])
    epoch = IntegerField(u'Epoch', validators=[InputRequired(), num_range(min = 1)])
    
    # training batch settings
    sub_batch_size = IntegerField(u'Training Sub-Batch Size', validators=[InputRequired(), num_range(min = 1, max = 32, dependant=[None, 'batch_size'])])
    val_batch_size = IntegerField(u'Val Batch Size', validators=[InputRequired(), num_range(min = 1, max = 32, dependant=[None, 'batch_size'])])
    
    # output
    output_name = StringField(u'Output File Name')
    output_compare = SelectMultipleField(u'Comparing with Other Models')
    
    submit = SubmitField('Start')
    
    def __init__(self, gpu_names):
        super(ResumeStreamForm, self).__init__()
        
        self.half_model.choices = [('', '')]
        # making sure the state and output files are all both existed for the models
        self.half_model.choices.extend([(x, x) for x in glob.glob(r'output\stream\**\training\*.pth.tar', recursive=True)])
        
        self._gpu_names = [('cuda:'+str(i), 'CUDA:'+str(i)+' '+gpu_names[i]) for i in range(len(gpu_names))]
        self._gpu_names.extend([('cpu', 'CPU')])
        self.device.choices = self._gpu_names
                
        #self.output_compare.choices = [(x, x) for x in os.listdir('output/stream/training') if '.pth.tar' in x]
        self.output_compare.choices = [(x,x) for x in glob.glob(r'output\stream\**\training\*.pth.tar', recursive=True)]
        
class TestStreamForm(FlaskForm):
    
    # basic setup
    full_model = SelectField(u'Select Model', validators=[InputRequired(), validate_empty_select])
    test_method = SelectField(u'Testing Method', choices=[('10-clips', '10-clips evenly distributed across video'), 
                                                          ('10-crops', '10-crops on each frames from each of 10 clips')], 
                                validators=[InputRequired()])
    device = SelectField(u'Device', choices = [], validators=[InputRequired()])
    
    # testing batch settings
    clip_len = IntegerField(u'Testing Clip Length', validators=[InputRequired(), num_range(min = 1)])
    test_batch_size = IntegerField(u'Testing Batch Size', validators=[InputRequired(), num_range(min = 1)], default = 1)
    test_subbatch_size = IntegerField(u'Subbatch Size (number of clips per f-prop)', validators=[InputRequired(), num_range(min = 1, max = 320, dependant=[None, '10x of test batch size'])], default = 10)
    
    # debugging mode
    is_debug_mode = BooleanField(u'Enable Debugging Mode')
    debug_mode = SelectField(u'Data Selection Mode', choices = [('peek','Peek (select first N samples regardless of label)'), ('distributed','Distributed (select first N samples across all labels)')], validators=[InputRequired()])
    debug_test_size = IntegerField(u'Test Set Size', default = 8, validators=[InputRequired()])
    
    submit = SubmitField('Start')
    
    def __init__(self, gpu_names):
        super(TestStreamForm, self).__init__()
        
        self.full_model.choices = [('', '')]
        # making sure the state and output files are all both existed for the models
        self.full_model.choices.extend([(x,'-'.join([x.split('.')[0].split('\\')[i] for i in [2, 3, 5]])) 
            for x in glob.glob(r'output\stream\**\training\*.pth.tar', recursive=True)])
        
        self._gpu_names = [('cuda:'+str(i), 'CUDA:'+str(i)+' '+gpu_names[i]) for i in range(len(gpu_names))]
        self._gpu_names.extend([('cpu', 'CPU')])
        self.device.choices = self._gpu_names
        
class InspectStreamForm(FlaskForm):
    
    # basic setup
    main_model = SelectField(u'Benchmarking Model', validators=[InputRequired(), validate_empty_select])
    model_compare = SelectMultipleField(u'Peer Models for Comparison (limit to 4)', validators=[selection_limit(4)])
    
    submit = SubmitField('Compute Result')
    
    def __init__(self):
        super(InspectStreamForm, self).__init__()
        
        self.main_model.choices = [('', '')]
        # making sure the state and output files are all both existed for the models
        self.main_model.choices.extend([(x,'-'.join([x.split('.')[0].split('\\')[i] for i in [2, 3, 5]]))
            for x in glob.glob('output\\stream\\**\\training\\*.pth.tar', recursive=True)])
        
        self.model_compare.choices = self.main_model.choices[1:]
        
class VisualizeStreamForm1(FlaskForm):
    
    # basic setup
    vis_model = SelectField(u'Stream Model', validators=[InputRequired(), validate_empty_select])
    device = SelectField(u'Device', choices = [], validators=[InputRequired()])
    
    submit = SubmitField('Load Model')
    
    def __init__(self, gpu_names):
        super(VisualizeStreamForm1, self).__init__()
        
        self.vis_model.choices = [('', '')]
        self.vis_model.choices.extend([(x,'-'.join([x.split('.')[0].split('\\')[i] for i in [2, 3, 5]]))
            for x in glob.glob('output\\stream\\**\\training\\*.pth.tar', recursive=True)])
        
        self.device.choices = [('cuda:'+str(i), 'CUDA:'+str(i)+' '+gpu_names[i]) for i in range(len(gpu_names))]
        self.device.choices.extend([('cpu', 'CPU')])
        
def validate_clip_name(form, field):
    if field.data not in form.video_list:
        raise ValidationError('Clip not found.')

class VisualizeStreamForm2(FlaskForm):
    
    clip_len = IntegerField(u'Clip Length', validators=[InputRequired(), num_range(min = 1)], default = 8)
    target_class = SelectField(u'Target Class', validators=[InputRequired()], coerce=int)
    clip_name = StringField(u'Testing Clip Name', validators=[InputRequired(), validate_clip_name], default = 'v_ApplyEyeMakeup_g01_c01')
    vis_layer = SelectMultipleField(u'GradCAM Target Layer', validators=[InputRequired()], coerce=str)
    
    submit = SubmitField('Load Visualizer')
    
    def __init__(self, clip_len, index_path, net_layer):
        super(VisualizeStreamForm2, self).__init__()
        
        f_in = open(index_path)
        self.target_class.choices = [(int(x.split(' ')[0]) - 1, x) for x in (f_in.read().split('\n'))[:-1]]
        f_in.close()
        
        self.video_list = os.listdir(os.path.join(index_path, '..', 'rgb'))
        
        self.clip_len.default = clip_len
        
        self.vis_layer.choices = [(x, x) for x in net_layer]
