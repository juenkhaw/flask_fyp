# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:37:09 2019

@author: Juen
"""

from __init__ import app, BASE_CONF
from flask import render_template, url_for, request, jsonify, redirect

import plotly.offline as py
import plotly.graph_objs as go

import torch

from threading import Thread

from forms import TrainStreamForm
from train_stream import StreamTrainer

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

def form_to_dict(form):
    a = [x for x in list(form.__class__.__dict__) if x[0] != '_' and 'validate' not in x and x != 'submit']
    return {x: getattr(form, x).data for x in a}

@app.route('/')
@app.route('/index')
def index():
        
    return render_template('index.html', base_meta = base_meta, device_info = device_info)

@app.route('/convert')
def convert():
    return "CONVERT"

@app.route('/train_stream', methods=['GET', 'POST'])
def train_stream():
    req = request.json
    print('train_stream/request.json', req)
    
    # check if this is ajax post from initializing page
    if (req == {} or req == None): # this is not ajax request
        # if stream trainer is not yet instantiated
        if 'st' not in globals():
            global st
            st = None
            
        # if stream trainer is None
        if st == None:
            base_meta['title'] = 'Stream Training'
            base_meta['header'] = 'Stream Training Setup'
            form = TrainStreamForm(device_info['gpu_names'])
            # start the stream training thread
            if form.validate_on_submit():
                st = StreamTrainer()
                Thread(target = (lambda: st.init(form_to_dict(form)))).start()
                #st = StreamTrainer(form_to_dict(form))
                return render_template('initializing.html', st = {'state' : st.state['INIT']})
            return render_template('train_stream_setup.html', base_meta = base_meta, form = form)
    
        else: # there is already an istance of train streamer
            if st.state['INIT']: # if st is ready
                return 'TRAINING START'
            else:
                return render_template('initializing.html', st = {'state' : st.state['INIT']})
        
    else: # this is ajax post request
        # if initialization is done
        print('train_stream/st/state/INIT', st.state['INIT'])
        if st.state['INIT']:
            return 'Training Start'
        else: # else do nothing
            return ''
        

@app.route('/test_stream')
def test_stream():
    return "TEST_STREAM"

@app.route('/inspect_stream')
def inspect_stream():
    return "INSPECT_STREAM"