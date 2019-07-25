# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:23:35 2019

@author: Juen
"""

from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, DecimalField, SubmitField
from wtforms.validators import DataRequired, regexp

import os

class TrainStreamForm(FlaskForm):
    modality = SelectField(u'Modality', choices = [('rgb','RGB'), ('flow','FLOW')], validators=[DataRequired()])
    dataset_path = StringField(u'Dataset Path', validators=[DataRequired()])
    base_lr = DecimalField(u'Base Learning Rate', places = None, validators=[DataRequired()])
    submit = SubmitField('Start')
    
    def validate_dataset_path(form, )