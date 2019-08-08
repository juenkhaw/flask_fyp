# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:37:09 2019

@author: Juen
"""

from . import app, BASE_CONFIG
from flask import render_template, url_for, request, jsonify, redirect

import torch

from threading import Thread
from os import listdir, path
from glob import glob
from time import sleep

from .forms import TrainStreamForm, num_range, SelectField, ResumeStreamForm, TestStreamForm, InspectStreamForm
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
    base_meta = {
        'title' : 'two-stream action recognition benchmarking', 
        'header' : 'Index'
        }
    return render_template('index.html', base_meta = base_meta, device_info = device_info)

def exec_train_stream(st):
    st.setup_training()
    st.train_net()
    st.update['complete'] = True

@app.route('/train_stream', methods=['GET', 'POST'])
def train_stream():
    req = request.json
    print('train_stream/request.json', req)
    
    # check if this is ajax post from initializing page
    if (req == {} or req == None): # this is not ajax request
        # if stream trainer is not yet instantiated
        if 'st' not in globals():
            print('train_stream/OH NO YOU SEE ME!')
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
                return render_template('initializing.html', url=url_for('train_stream'))
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
                return render_template('train_net_state.html', base_meta=base_meta, st=st, url=url_for('train_stream'))
            else:
                return render_template('initializing.html', url=url_for('train_stream'))
        
    else: # this is ajax post request, classifying action based on request json
        
        if all(k in req.keys() for k in ['dataset', 'split']):
            read_path = 'output/stream/'+req['dataset']+'/split'+req['split']+'/training'
            if not path.exists(read_path):
                form.output_compare.choices = []
            else:
                form.output_compare.choices = [(x, x) for x in listdir(read_path) if '.pth.tar' in x]
            return jsonify({'html': form.output_compare})
        
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
            if 'update' in req.keys():
                if st.update['init']:
                    response = {'progress':st.update['progress'], 'result':st.update['result'], 'complete':st.update['complete'], 'no_compare':(st.compare == [])}
                    print(response)
                    
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
                        st.update['complete'] = False
                        response['progress_html'] = render_template('train_progress_div.html', st=st)
                        del globals()['st']
                        torch.cuda.empty_cache()
                        
                    return jsonify(response)
                    
                else: # else do nothing
                    return ''
            else: # for initializing request
                if st.update['init']:
                    return 'done'
                else:
                    return ''
    
def exec_resume_stream(st):
    st.setup_resume_training()
    st.train_net()
    st.update['complete'] = True

@app.route('/resume_stream', methods=['GET', 'POST'])
def resume_stream():
    req = request.json
    print('resume_stream/request.json', req)
    
    # check if this is ajax post from initializing page
    if (req == {} or req == None): # this is not ajax request
        # if stream trainer is not yet instantiated
        if 'st' not in globals():
            print('resume_stream/OH NO YOU SEE ME!')
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
                print('resume_stream/form_success')
                st = StreamTrainer(form_to_dict(form))
                Thread(target = exec_resume_stream, args = (st,)).start()
                return render_template('initializing.html', url=url_for('resume_stream'))
                #return jsonify(form_to_dict(form))
            else:
                for fieldName, errorMessages in form.errors.items():
                    for err in errorMessages:
                        print('resume_stream/form_failed',fieldName, err)
            return render_template('resume_stream_setup.html', base_meta = base_meta, form = form)
        
        else: # there is already an istance of train streamer
            if st.update['init']: # if st is ready
                #return 'TRAINING RESUME'
                base_meta['title'] = 'resume_stream/'+st.args['output_name']
                base_meta['header'] = 'Resume Stream Training - '+st.args['output_name']
                return render_template('train_net_state.html', base_meta=base_meta, st=st, url=url_for('resume_stream'))
            else:
                return render_template('initializing.html', url=url_for('resume_stream'))
            
    else: # there is something in the request json
        
        if 'model' in req.keys(): # update properties and forms default value
            if req['model'] == '':
                return jsonify({'html':'<span style="color: red;">No data available</span>'})
            else:
                p = torch.load(req['model'], map_location=lambda storage, loc: storage)
                p_args = p['args']
                p_training = {x:p['output'][x] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}
                p_epoch = p['epoch']
                del p
                form.epoch.validators[1] = num_range(min=p_epoch, dependant=['previous epoch', None])
                form.sub_batch_size.validators[1] = num_range(min = 1, max = p_args['batch_size'], dependant=[None,'batch size'])
                form.val_batch_size.validators[1] = num_range(min = 1, max = p_args['batch_size'], dependant=[None,'batch size'])
                form.output_compare.choices = [x for x in form.half_model.choices if x[0] != req['model'] and x[0] != '']
                return jsonify({'html':render_template('resume_stream_properties.html', args = p_args, training = p_training, epoch = p_epoch), 
                                'device':p_args['device'], 'sub':p_args['sub_batch_size'], 
                                'val':p_args['val_batch_size'], 'name':p_args['output_name'], 
                                'compare_html':form.output_compare, 'compare':p_args['output_compare']})
        else: # if initialization is done
            #print('train_stream/st/update/init', st.update['init'])
            if 'update' in req.keys(): # for updating request
                if st.update['init']:
                    response = {'progress':st.update['progress'], 'result':st.update['result'], 'complete':st.update['complete'], 'no_compare':(st.compare == [])}
                    print(response)
                    
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
                        st.update['complete'] = False
                        response['progress_html'] = render_template('train_progress_div.html', st=st)
                        del globals()['st']
                        torch.cuda.empty_cache()
                        
                    return jsonify(response)
                else:
                    return ''
                
            else: # for initializing request
                if st.update['init']:
                    return 'done'
                else:
                    return ''
        
def exec_test_stream(st):
    st.setup_testing()
    st.test_net()
    st.update['complete'] = True

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
                #Thread(target = (lambda: st.init(form_to_dict(form)))).start()
                st = StreamTrainer(form_to_dict(form))
                print('test_stream/form_success')
                # retrieve training state graph
                p = torch.load(form.full_model.data, map_location=lambda storage, loc: storage)
                main_loss_graph(p['epoch'], p['output']['train_loss'], p['output']['val_loss'])
                main_acc_graph(p['epoch'], p['output']['train_acc'], p['output']['val_acc'])
                Thread(target = exec_test_stream, args = (st,)).start()
                #return jsonify(form_to_dict(form))
                return render_template('initializing.html', url=url_for('test_stream'))
            else:
                for fieldName, errorMessages in form.errors.items():
                    for err in errorMessages:
                        print('test_stream/form_failed',fieldName, err)
            return render_template('test_stream_setup.html', base_meta = base_meta, form = form)
        
        else:
            if st.update['init']:
                base_meta['title'] = 'test_stream/'+st.args['output_name']
                base_meta['header'] = 'Stream Evaluation - '+st.args['output_name']
                return render_template('test_stream_state.html', base_meta=base_meta, st=st, 
                                       no_val=BASE_CONFIG['dataset'][st.args['dataset']]['val_txt'] == [])
            else:
                return render_template('initializing.html', url=url_for('test_stream'))
            
    else: # there is something in the request json
        if 'model' in req.keys(): # update properties and forms default value
            if req['model'] == '':
                return jsonify({'html':'<span style="color: red;">No data available</span>'})
            else:
                p = torch.load(req['model'], map_location=lambda storage, loc: storage)
                p_args = p['args']
                p_training = {x:p['output'][x] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}
                p_epoch = p['epoch']
                del p
                return jsonify({'html':render_template('resume_stream_properties.html', args = p_args, training = p_training, epoch = p_epoch), 
                                'device': p_args['device'], 'clip_len':p_args['clip_len'], 'debug':p_args['is_debug_mode']})
    
        elif 'batch' in req.keys(): # update subbatch size validator
            form.test_subbatch_size.validators[1] = num_range(min = 1, max = int(req['batch']) * 10, dependant=[None, '10x of test batch size'])
            return ''
        
        else: # if initialization is done
            #print('train_stream/st/update/init', st.update['init'])
            if 'update' in req.keys():
                if st.update['init']:
                    response = {'progress':st.update['progress'], 'complete':st.update['complete']}
                    print(response)
                    
                    if st.update['progress']: # update progress status
                        st.update['progress'] = False
                        response['progress_html'] = render_template('test_progress_div.html', st=st)
                        
                    if st.update['complete']:
                        st.update['complete'] = False
                        del globals()['st']
                        torch.cuda.empty_cache()
                        
                    return jsonify(response)
                    
                else: # else do nothing
                    return ''
            else: # for initializing request
                if st.update['init']:
                    return 'done'
                else:
                    return ''

@app.route('/inspect_stream', methods=['GET', 'POST'])
def inspect_stream():
    req = request.json
    print('inspect_stream/request.json', req)
    
    if (req == {} or req == None): # this is not ajax request
        base_meta['title'] = 'stream/inspect_stream'
        base_meta['header'] = 'Stream Output Viewer'
        global form
        form = InspectStreamForm()
        
        if form.validate_on_submit():
            print('inspect_stream/form_success')
            base_meta['title'] = 'inspect_stream/' + form.main_model.data
            base_meta['header'] = 'Stream Viewer - ' + form.main_model.data
            
            # ack models that need to be read
            if form.main_model.data in form.model_compare.data:
                form.model_compare.data.remove(form.main_model.data)
            
            # extract basic info on base model
            temp = form.main_model.data.split('\\')
            main_model = temp[-1].split('.')[0]
            base_dataset = temp[2]
            base_split = temp[3]
            
            # appending url of peer pkgs
            model_url_list = [form.main_model.data]
            model_url_list.extend(form.model_compare.data)
            peer_model = [x.split('\\')[-1].split('.')[0] for x in model_url_list]
                        
            test_method = ['10-clips', '10-crops']
            
            # initialize response
            output = {model_name : {'test_result' : {method : None for method in test_method}} for model_name in peer_model}
            
            # read training package details
            train_pkg = {url.split('\\')[-1].split('.')[0] : torch.load(url, map_location=lambda storage, loc: storage) for url in model_url_list}
                        
            # read testing state of each pkgs
            test_url = glob('output\\stream\\'+base_dataset+'\\'+base_split+'\\testing\\*.pth.tar')
            filtered_test_url = {}
            for url in test_url:
                test_pkg_name = '_'.join(url.split('\\')[-1].split('_')[:-1])
                test_pkg_method = url.split('\\')[-1].split('.')[0].split('_')[-1]
                if test_pkg_name in peer_model and test_pkg_method in test_method:
                    filtered_test_url.update({test_pkg_name+'=/='+test_pkg_method : url})
                    
            # reading results
            test_pkg = {k : torch.load(url, map_location=lambda storage, loc: storage) for k, url in filtered_test_url.items()}
            
            for name, pkg in test_pkg.items():
                model_method = name.split('=/=')[1]
                model_name = name.split('=/=')[0]
                for method in test_method:
                    if method == model_method:
                        for k, v in pkg['acc'].items():
                            pkg['acc'][k] = round(v * 100, 2)
                        output[model_name]['test_result'][method] = pkg['acc']
                        
            # training state
            main_loss_graph(train_pkg[main_model]['epoch'], train_pkg[main_model]['output']['train_loss'], train_pkg[main_model]['output']['val_loss'], title=main_model)
            main_acc_graph(train_pkg[main_model]['epoch'], train_pkg[main_model]['output']['train_acc'], train_pkg[main_model]['output']['val_acc'], title=main_model)
            if len(peer_model) > 1: # if there is any comparison
                comparing_graph(train_pkg[model_name], [model for k, model in train_pkg.items() if k != model_name], 'train_loss', 'loss')
                comparing_graph(train_pkg[model_name], [model for k, model in train_pkg.items() if k != model_name], 'val_loss', 'loss')
                comparing_graph(train_pkg[model_name], [model for k, model in train_pkg.items() if k != model_name], 'train_acc', 'accuracy')
                comparing_graph(train_pkg[model_name], [model for k, model in train_pkg.items() if k != model_name], 'val_acc', 'accuracy')
            
            # argument filters
            args_filter = ['modality', 'dataset', 'split', 'network', 'pretrain_model', 'freeze_point', 'base_lr', 
                           'batch_size', 'momentum', 'l2decay', 'dropout', 'lr_scheduler', 'lr_reduce_ratio', 
                           'step_size', 'last_step', 'patience', 'loss_threshold', 'min_lr', 'clip_len', 
                           'is_mean_sub', 'is_rand_flip']
            for name, pkg in train_pkg.items():
                output[name]['args'] = {arg : train_pkg[name]['args'][arg] for arg in args_filter}
                output[name]['args']['epoch'] = pkg['epoch']
            
            
            return render_template('inspect_stream.html', base_meta=base_meta, output=output, base_model=main_model)
        else:
            for fieldName, errorMessages in form.errors.items():
                for err in errorMessages:
                    print('inspect_stream/form_failed',fieldName, err)
        
        return render_template('inspect_stream_setup.html', base_meta=base_meta, form=form)
    
    else: # ajax request
        if 'base_model' in req.keys():
            base_dataset = req['base_model'].split('\\')[2]
            base_split = req['base_model'].split('\\')[3]
            form.model_compare.choices = [(x,x) for x in glob('output\\stream\\'+base_dataset+'\\'+base_split+'\\training\\*.pth.tar', recursive=True)]
            return jsonify({'html':form.model_compare})
        
        else:
            return ''