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

class TrainStreamForm(FlaskForm):
    
    _gpu_names = []
    _split_choices = []
    
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
        self.pretrain_model.choices.extend([(x, x) for x in os.listdir('pretrained') if '.pth.tar' in x])
        
        self.output_compare.choices = [(x, x) for x in os.listdir('output/stream/training') if '.pth.tar' in x]
        
        self.freeze_point.choices = [('none', 'None')]
        for net in BASE_CONFIG['network'].keys():
            self.freeze_point.choices.extend([(x, x) for x in BASE_CONFIG['network'][net]['endpoint']])
    
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
        state_list = os.listdir('output/stream/state')
        self.half_model.choices.extend([(x, x) for x in os.listdir('output/stream/training') if '.pth.tar' in x and x in state_list])
        
        self._gpu_names = [('cuda:'+str(i), 'CUDA:'+str(i)+' '+gpu_names[i]) for i in range(len(gpu_names))]
        self._gpu_names.extend([('cpu', 'CPU')])
        self.device.choices = self._gpu_names
                
        self.output_compare.choices = [(x, x) for x in os.listdir('output/stream/training') if '.pth.tar' in x]
        
class TestStreamForm(FlaskForm):
    
    # basic setup
    full_model = SelectField(u'Select Model', validators=[InputRequired(), validate_empty_select])
    test_method = SelectField(u'Testing Method', choices=[('10-clips', '10-clips evenly distributed across video'), 
                                                          ('10-crops', '10-crops on frames at the middle of video')], 
                                validators=[InputRequired()])
    device = SelectField(u'Device', choices = [], validators=[InputRequired()])
    
    # testing batch settings
    clip_len = IntegerField(u'Testing Clip Length', validators=[InputRequired(), num_range(min = 1)])
    test_batch_size = IntegerField(u'Testing Batch Size (number of samples with 10 clips each)', validators=[InputRequired(), num_range(min = 1)], default = 32)
    test_subbatch_size = IntegerField(u'Subbatch Size (number of clips per f-prop)', validators=[InputRequired(), num_range(min = 1, max = 320, dependant=[None, '10x of test batch size'])], default = 320)
    
    # debugging mode
    is_debug_mode = BooleanField(u'Enable Debugging Mode')
    debug_mode = SelectField(u'Data Selection Mode', choices = [('peek','Peek (select first N samples regardless of label)'), ('distributed','Distributed (select first N samples across all labels)')], validators=[InputRequired()])
    debug_test_size = IntegerField(u'Test Set Size', default = 32, validators=[InputRequired()])
    
    submit = SubmitField('Start')
    
    def __init__(self, gpu_names):
        super(TestStreamForm, self).__init__()
        
        self.full_model.choices = [('', '')]
        # making sure the state and output files are all both existed for the models
        state_list = os.listdir('output/stream/state')
        self.full_model.choices.extend([(x, x) for x in os.listdir('output/stream/training') if '.pth.tar' in x and x in state_list])
        
        self._gpu_names = [('cuda:'+str(i), 'CUDA:'+str(i)+' '+gpu_names[i]) for i in range(len(gpu_names))]
        self._gpu_names.extend([('cpu', 'CPU')])
        self.device.choices = self._gpu_names