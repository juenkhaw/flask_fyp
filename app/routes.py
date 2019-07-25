# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:37:09 2019

@author: Juen
"""

from app import app
from flask import render_template, url_for, request, jsonify

import plotly.offline as py
import plotly.graph_objs as go

import torch

from app.forms import TrainStreamForm

graph_config = {'scrollZoom' : False, 'displayModeBar' : False}

base_meta = {
        'title' : 'Action Recognition Network Benchmarking', 
        'header' : 'Main Page'
}

device_info = {
    'cuda_support' : torch.cuda.is_available(),
    'gpu_count' : torch.cuda.device_count(),
    'gpu_names' : [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
}

@app.route('/')
@app.route('/index')
def index():
        
    return render_template('index.html', base_meta = base_meta, device_info = device_info)

@app.route('/convert')
def convert():
    return "CONVERT"

@app.route('/train_stream', methods=['GET', 'POST'])
def train_stream():
    base_meta['title'] = 'Stream Training'
    base_meta['header'] = 'Stream Training Setup'
    form = TrainStreamForm()
    if form.validate_on_submit():
        return form.modality.data + '/' + form.dataset_path.data + '/' + str(form.base_lr.data)
    return render_template('train_stream_setup.html', base_meta = base_meta, device_info = device_info, form = form)

@app.route('/test_stream')
def test_stream():
    return "TEST_STREAM"

@app.route('/inspect_stream')
def inspect_stream():
    return "INSPECT_STREAM"