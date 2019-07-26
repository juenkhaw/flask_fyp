# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:23:35 2019

@author: Juen
"""

from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, DecimalField, SubmitField, ValidationError, FileField
from wtforms.validators import DataRequired, regexp

from __init__ import BASE_CONF

import os

class TrainStreamForm(FlaskForm):
            
    modality = SelectField(u'Modality', choices = [('rgb','RGB'), ('flow','FLOW')], validators=[DataRequired()])
    dataset = SelectField(u'Video Dataset', choices = [(x, x) for x in list(BASE_CONF['dataset'].keys())], validators=[DataRequired()])
    base_lr = DecimalField(u'Base Learning Rate', places = None, default = 0.01)
    network = SelectField(u'Network Architecture', choices = [(x, x) for x in list(BASE_CONF['network'].keys())], validators=[DataRequired()])
    #network_path = StringField(u'Network py File Name', validators=[DataRequired()])
    submit = SubmitField('Start')
    
    def validate_dataset_path(form, field):
        try:
            dirs = os.listdir(field.data)
        except:
            raise ValidationError('The path specified could not be read')
        
        if not all(x in dirs for x in ['rgb', 'flow']) :
            raise ValidationError('The path specified does not contain rgb/flow folder')
