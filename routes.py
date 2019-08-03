# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:37:09 2019

@author: Juen
"""

from . import app, BASE_CONFIG
from flask import render_template, url_for, request, jsonify, redirect

import torch

from threading import Thread
from os import listdir
from time import sleep

from .forms import TrainStreamForm, num_range, SelectField, ResumeStreamForm, TestStreamForm
from decimal import Decimal

from .stream_module import StreamTrainer

from .graph import *

graph_config = {'scrollZoom' : False, 'displayModeBar' : False}

base_meta = {
        'title' : 'two-stream action recognition benchmarking', 
        'header' : 'Index'
}

device_info = {
    'cuda_support' : torch.cuda.is_available(),
    'gpu_count' : torch.cuda.device_count(),
    'gpu_names' : [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
}

def form_to_dict(form):
    a = [x for x in list(form.__class__.__dict__) if x[0] != '_' and 'validate' not in x and x != 'submit']
    dic = {}
    for field_name in a:
        field = getattr(form, field_name)
        if type(field) == SelectField:
            if field_name in ['split']:
                dic[field_name] = field.data
            else:
                dic[field_name] = str(field.data)
        elif type(field.data) == Decimal:
            dic[field_name] = float(field.data)
        else:
            dic[field_name] = field.data
    return dic

@app.route('/')
@app.route('/index')
def index():   
    return render_template('index.html', base_meta = base_meta, device_info = device_info)

def exec_train_stream(st):
    st.setup_training()
    st.train_net()
    st.update['complete'] = True

@app.route('/train_stream', methods=['GET', 'POST'])
def train_stream():
    req = request.json
    #print('train_stream/request.json', req)
    
    # check if this is ajax post from initializing page
    if (req == {} or req == None): # this is not ajax request
        # if stream trainer is not yet instantiated
        if 'st' not in globals():
            global st
            st = None
            
        # if stream trainer is None
        if st == None:
            base_meta['title'] = 'stream/train_stream'
            base_meta['header'] = 'New Stream Training'
            global form
            form = TrainStreamForm(device_info['gpu_names'])
            # start the stream training thread
            if form.validate_on_submit():
                #Thread(target = (lambda: st.init(form_to_dict(form)))).start()
                st = StreamTrainer(form_to_dict(form))
                print('train_stream/form_success')
                Thread(target = exec_train_stream, args = (st,)).start()
                #return jsonify(form_to_dict(form))
                return render_template('initializing.html')
            else:
                for fieldName, errorMessages in form.errors.items():
                    for err in errorMessages:
                        print('train_stream/form_failed',fieldName, err)
            return render_template('train_stream_setup.html', base_meta = base_meta, form = form)
    
        else: # there is already an istance of train streamer
            if st.update['init']: # if st is ready
                #return 'TRAINING START'
                base_meta['title'] = 'train_stream/'+st.args['output_name']
                base_meta['header'] = 'Stream Training - '+st.args['output_name']
                return render_template('train_net_state.html', base_meta=base_meta, st=st)
            else:
                return render_template('initializing.html')
        
    else: # this is ajax post request, classifying action based on request json
        
        if 'dataset' in req.keys(): # updating the split_select choices
            form.split.choices = [(x, x) for x in list(range(1, BASE_CONFIG['dataset'][req['dataset']]['split'] + 1))]
            return jsonify({'html':form.split})
        
        elif all(x in req.keys() for x in ['batchsize', 'field']): # updating max value for sub-batch and val-batch
            if req['field'] == 'sub':
                form.sub_batch_size.validators[1] = num_range(min = 1, max = int(req['batchsize']), dependant=[None,'batch size'])
                return jsonify({'html':form.sub_batch_size})
            else:
                form.val_batch_size.validators[1] = num_range(min = 1, max = int(req['batchsize']), dependant=[None,'batch size'])
                return jsonify({'html':form.val_batch_size})
            
        elif 'base_lr' in req.keys(): # updating max value for min_lr in reduce_lr_on_plateau settings
            form.min_lr.validators[1] = num_range(min = 0, max = float(req['base_lr']))
            return jsonify({'html':form.min_lr})
        
        elif 'network' in req.keys(): # updating freeze_point options
            form.freeze_point.choices = [('none', 'None')]
            form.freeze_point.choices.extend([(x,x) for x in BASE_CONFIG['network'][req['network']]['endpoint']])
            return jsonify({'html':form.freeze_point})
            
        else: # if initialization is done
            #print('train_stream/st/update/init', st.update['init'])
            if st.update['init']:
                response = {'progress':st.update['progress'], 'result':st.update['result'], 'complete':st.update['complete'], 'no_compare':(st.compare == [])}
                
                if st.update['progress']: # update progress status
                    st.update['progress'] = False
                    response['progress_html'] = render_template('train_progress_div.html', st=st)
                    
                if st.update['result']: # update result
                    st.update['result'] = False
                    main_loss_graph(st.epoch, st.main_output['output']['train_loss'], st.main_output['output']['val_loss'])
                    main_acc_graph(st.epoch, st.main_output['output']['train_acc'], st.main_output['output']['val_acc'])
                    if st.compare != []:
                        comparing_graph(st.main_output, st.compare, 'train_loss', 'loss')
                        comparing_graph(st.main_output, st.compare, 'val_loss', 'loss')
                        comparing_graph(st.main_output, st.compare, 'train_acc', 'accuracy')
                        comparing_graph(st.main_output, st.compare, 'val_acc', 'accuracy')
                    
                if st.update['complete']:
                    response['progress_html'] = render_template('train_progress_div.html', st=st)
                    st = None
                    
                return jsonify(response)
                
            else: # else do nothing
                return ''
    
def exec_resume_stream(st):
    st.setup_resume_training()
    st.train_net()

@app.route('/resume_stream', methods=['GET', 'POST'])
def resume_stream():
    req = request.json
    print('resume_stream/request.json', req)
    
    # check if this is ajax post from initializing page
    if (req == {} or req == None): # this is not ajax request
        # if stream trainer is not yet instantiated
        if 'st' not in globals():
            global st
            st = None
            
        # if stream trainer is None
        if st == None:
            base_meta['title'] = 'stream/resume_stream'
            base_meta['header'] = 'Resume Stream Training'
            global form
            form = ResumeStreamForm(device_info['gpu_names'])
            # start the stream training thread
            if form.validate_on_submit():
                #Thread(target = (lambda: st.init(form_to_dict(form)))).start()
                st = StreamTrainer(form_to_dict(form))
                #Thread(target = exec_resume_stream, args = (st,)).start()
                print('resume_stream/form_success')
                #return render_template('initializing.html')
                return jsonify(form_to_dict(form))
            else:
                for fieldName, errorMessages in form.errors.items():
                    for err in errorMessages:
                        print('resume_stream/form_failed',fieldName, err)
            return render_template('resume_stream_setup.html', base_meta = base_meta, form = form)
        
        else: # there is already an istance of train streamer
            if st.update['init']: # if st is ready
                return 'TRAINING RESUME'
            else:
                return render_template('initializing.html')
            
    else: # there is something in the request json
        
        if 'model' in req.keys(): # update properties and forms default value
            if req['model'] == '':
                return jsonify({'html':'<span style="color: red;">No data available</span>'})
            else:
                p = torch.load('output/stream/training/'+req['model'], map_location=lambda storage, loc: storage)
                p_args = p['args']
                p_training = {x:p['output'][x] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}
                p_epoch = p['epoch']
                del p
                form.epoch.validators[1] = num_range(min=p_epoch, dependant=['previous epoch', None])
                form.sub_batch_size.validators[1] = num_range(min = 1, max = p_args['batch_size'], dependant=[None,'batch size'])
                form.val_batch_size.validators[1] = num_range(min = 1, max = p_args['batch_size'], dependant=[None,'batch size'])
                form.output_compare.choices = [(x, x) for x in listdir('output/stream/training') if '.pth.tar' in x and x != p_args['output_name']+'.pth.tar']
                return jsonify({'html':render_template('resume_stream_properties.html', args = p_args, training = p_training, epoch = p_epoch), 
                                'device':p_args['device'], 'sub':p_args['sub_batch_size'], 
                                'val':p_args['val_batch_size'], 'name':p_args['output_name'], 
                                'compare_html':form.output_compare, 'compare':p_args['output_compare']})
        else: # if initialization is done
            print('resume_stream/st/update/init', st.update['init'])
            if st.update['init']:
                return 'Training Resume'
            else: # else do nothing
                return ''
        
def exec_test_stream(st):
    st.setup_testing()
    st.test_net()

@app.route('/test_stream', methods=['GET', 'POST'])
def test_stream():
    req = request.json
    print('test_stream/request.json', req)
    
    # check if this is ajax post from initializing page
    if (req == {} or req == None): # this is not ajax request
        # if stream trainer is not yet instantiated
        if 'st' not in globals():
            global st
            st = None
            
        # if stream trainer is None
        if st == None:
            base_meta['title'] = 'stream/test_stream'
            base_meta['header'] = 'Stream Evaluation'
            global form
            form = TestStreamForm(device_info['gpu_names'])
            # start the stream training thread
            if form.validate_on_submit():
                #st = StreamTrainer()
                #Thread(target = (lambda: st.init(form_to_dict(form)))).start()
                #st = StreamTrainer(form_to_dict(form))
                print('test_stream/form_success')
                return jsonify(form_to_dict(form))
            else:
                for fieldName, errorMessages in form.errors.items():
                    for err in errorMessages:
                        print('test_stream/form_failed',fieldName, err)
            return render_template('test_stream_setup.html', base_meta = base_meta, form = form)
            
    else: # there is something in the request json
        if 'model' in req.keys(): # update properties and forms default value
            if req['model'] == '':
                return jsonify({'html':'<span style="color: red;">No data available</span>'})
            else:
                p = torch.load('output/stream/training/'+req['model'], map_location=lambda storage, loc: storage)
                p_args = p['args']
                p_training = {x:p['output'][x] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}
                p_epoch = p['epoch']
                del p
                return jsonify({'html':render_template('resume_stream_properties.html', args = p_args, training = p_training, epoch = p_epoch), 
                                'device': p_args['device'], 'clip_len':p_args['clip_len'], 'debug':p_args['is_debug_mode']})
    
        elif 'batch' in req.keys(): # update subbatch size validator
            form.test_subbatch_size.validators[1] = num_range(min = 1, max = int(req['batch']) * 10, dependant=[None, '10x of test batch size'])
            return ''

@app.route('/inspect_stream')
def inspect_stream():
    return "INSPECT_STREAM"