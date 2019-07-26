# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:23:35 2019

@author: Juen
"""

from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, DecimalField, SubmitField, ValidationError, FileField, IntegerField
from wtforms.validators import DataRequired, regexp

from __init__ import BASE_CONF

import os

def num_range(min = None, max = None):
    errors = ['Must not be less than ' + str(min), 'Must not be more than ' + str(max)]
    
    def _num_range(form, field):
        if min != None:
            if field.data < min:
                raise ValidationError(errors[0])
        if max != None:
            if field.data > max:
                raise ValidationError(errors[1])
                
    return _num_range

class TrainStreamForm(FlaskForm):
    
    gpu_names = []
    split_choices = []
    
    # basic setup
    modality = SelectField(u'Modality', choices = [('rgb','RGB'), ('flow','FLOW')], validators=[DataRequired()])
    dataset = SelectField(u'Video Dataset', choices = [(x, x) for x in list(BASE_CONF['dataset'].keys())], validators=[DataRequired()])
    split = SelectField(u'Split', choices = [], validators=[DataRequired()], id = 'split_select')
    network = SelectField(u'Network Architecture', choices = [(x, x) for x in list(BASE_CONF['network'].keys())], validators=[DataRequired()])
    device = SelectField(u'Device', choices = [], validators=[DataRequired()])
    
    # optimizer
    base_lr = DecimalField(u'Base Learning Rate', places = None, default = 0.01)
    
    # data preprocessing
    clip_len = IntegerField(u'Clip Length', validators=[DataRequired(), num_range(min = 1)])
    
    # debugging mode
    
    # output

    submit = SubmitField('Start')
    
    def __init__(self, gpu_names):
        super(TrainStreamForm, self).__init__()
        self.gpu_names = [('cuda:'+str(i), 'CUDA:'+str(i)+' '+gpu_names[i]) for i in range(len(gpu_names))]
        self.gpu_names.extend([('cpu', 'CPU')])
        self.device.choices = self.gpu_names
        
        first_dataset = list(BASE_CONF['dataset'].keys())[0]
        self.split_choices = [(x, x) for x in list(range(1, BASE_CONF['dataset'][first_dataset]['split'] + 1))]
        self.split.choices = self.split_choices
    
    def validate_dataset_path(form, field):
        try:
            dirs = os.listdir(field.data)
        except:
            raise ValidationError('The path specified could not be read')
        
        if not all(x in dirs for x in ['rgb', 'flow']) :
            raise ValidationError('The path specified does not contain rgb/flow folder')
