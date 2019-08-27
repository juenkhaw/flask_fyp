# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:37:09 2019

@author: Juen
"""

from . import app, BASE_CONFIG
from flask import render_template, url_for, request, jsonify, redirect

import torch
import numpy as np

from threading import Thread
from os import listdir, path
from glob import glob
from time import sleep
from json import loads

from .forms import TrainStreamForm, num_range, SelectField, ResumeStreamForm, TestStreamForm, InspectStreamForm, VisualizeStreamForm1, VisualizeStreamForm2
from decimal import Decimal

from .stream_module import StreamTrainer
from .visualize_module import StreamVis

from .graph import *
import os

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
        'header' : 'Toolset for Two-Stream Action Classification Models'
        }
    return render_template('index.html', base_meta = base_meta, device_info = device_info)

def exec_train_stream(st):
    st.setup_training()
    st.train_net()
    st.update['init'] = False
    st.update['complete'] = True
    
def exec_resume_stream(st):
    st.setup_resume_training()
    st.train_net()
    st.update['init'] = False
    st.update['complete'] = True

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
            base_meta['title'] = 'Stream Trainer'
            base_meta['header'] = 'Stream Trainer'
            global form
            form = TrainStreamForm(device_info['gpu_names'])
            # start the stream training thread
            if form.validate_on_submit():
                #Thread(target = (lambda: st.init(form_to_dict(form)))).start()
                st = StreamTrainer(form_to_dict(form))
                print('train_stream/form_success')
                if form.half_model.data == '':
                    Thread(target = exec_train_stream, args = (st,)).start()
                else:
                    Thread(target = exec_resume_stream, args = (st,)).start()
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
                base_meta['title'] = 'Stream Trainer'
                base_meta['header'] = 'Stream Trainer - '+st.args['output_name']
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
        
        elif 'model' in req.keys():
            if req['model'] == '':
                form.epoch.validators[1] = num_range(min=1, dependant=[None, None])
                return jsonify({'pkg_html':'<span style="color: red;">No data available</span>', 'form_epoch':form.epoch})
            else:
                p = torch.load(req['model'], map_location=lambda storage, loc: storage)
                p_args = p['args']
                p_training = {x:p['output'][x] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}
                p_epoch = p['epoch']
                # update form object default value
                del p
                form.epoch.validators[1] = num_range(min=p_epoch + 1, dependant=['previous epoch', None])
                form_keys= list(form_to_dict(form).keys())
                resume_args = {k : p_args[k] for k in form_keys if k != 'half_model'}
                return jsonify({'pkg_html':render_template('resume_stream_properties.html', args = p_args, training = p_training, epoch = p_epoch), 
                                'resume_args':resume_args, 'form_epoch':form.epoch})
            
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
                            comparing_graph2(st.main_output, st.compare, 'Loss', 'loss')
                            comparing_graph2(st.main_output, st.compare, 'Accuracy', 'acc')
#                            comparing_graph(st.main_output, st.compare, 'train_loss', 'loss')
#                            comparing_graph(st.main_output, st.compare, 'val_loss', 'loss')
#                            comparing_graph(st.main_output, st.compare, 'train_acc', 'accuracy')
#                            comparing_graph(st.main_output, st.compare, 'val_acc', 'accuracy')
                        
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

@app.route('/resume_stream', methods=['GET', 'POST'])
def resume_stream():
    return 'Deprecated'
#    req = request.json
#    print('resume_stream/request.json', req)
#    
#    # check if this is ajax post from initializing page
#    if (req == {} or req == None): # this is not ajax request
#        # if stream trainer is not yet instantiated
#        if 'st' not in globals():
#            print('resume_stream/OH NO YOU SEE ME!')
#            global st
#            st = None
#            
#        # if stream trainer is None
#        if st == None:
#            base_meta['title'] = 'stream/resume_stream'
#            base_meta['header'] = 'Resume Stream Training'
#            global form
#            form = ResumeStreamForm(device_info['gpu_names'])
#            # start the stream training thread
#            if form.validate_on_submit():
#                #Thread(target = (lambda: st.init(form_to_dict(form)))).start()
#                print('resume_stream/form_success')
#                st = StreamTrainer(form_to_dict(form))
#                Thread(target = exec_resume_stream, args = (st,)).start()
#                return render_template('initializing.html', url=url_for('resume_stream'))
#                #return jsonify(form_to_dict(form))
#            else:
#                for fieldName, errorMessages in form.errors.items():
#                    for err in errorMessages:
#                        print('resume_stream/form_failed',fieldName, err)
#            return render_template('resume_stream_setup.html', base_meta = base_meta, form = form)
#        
#        else: # there is already an istance of train streamer
#            if st.update['init']: # if st is ready
#                #return 'TRAINING RESUME'
#                base_meta['title'] = 'resume_stream/'+st.args['output_name']
#                base_meta['header'] = 'Resume Stream Training - '+st.args['output_name']
#                return render_template('train_net_state.html', base_meta=base_meta, st=st, url=url_for('resume_stream'))
#            else:
#                return render_template('initializing.html', url=url_for('resume_stream'))
#            
#    else: # there is something in the request json
#        
#        if 'model' in req.keys(): # update properties and forms default value
#            if req['model'] == '':
#                return jsonify({'html':'<span style="color: red;">No data available</span>'})
#            else:
#                p = torch.load(req['model'], map_location=lambda storage, loc: storage)
#                p_args = p['args']
#                p_training = {x:p['output'][x] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}
#                p_epoch = p['epoch']
#                del p
#                form.epoch.validators[1] = num_range(min=p_epoch, dependant=['previous epoch', None])
#                form.sub_batch_size.validators[1] = num_range(min = 1, max = p_args['batch_size'], dependant=[None,'batch size'])
#                form.val_batch_size.validators[1] = num_range(min = 1, max = p_args['batch_size'], dependant=[None,'batch size'])
#                form.output_compare.choices = [x for x in form.half_model.choices if x[0] != req['model'] and x[0] != '']
#                return jsonify({'html':render_template('resume_stream_properties.html', args = p_args, training = p_training, epoch = p_epoch), 
#                                'device':p_args['device'], 'sub':p_args['sub_batch_size'], 
#                                'val':p_args['val_batch_size'], 'name':p_args['output_name'], 
#                                'compare_html':form.output_compare, 'compare':p_args['output_compare']})
#        else: # if initialization is done
#            #print('train_stream/st/update/init', st.update['init'])
#            if 'update' in req.keys(): # for updating request
#                if st.update['init']:
#                    response = {'progress':st.update['progress'], 'result':st.update['result'], 'complete':st.update['complete'], 'no_compare':(st.compare == [])}
#                    print(response)
#                    
#                    if st.update['progress']: # update progress status
#                        st.update['progress'] = False
#                        response['progress_html'] = render_template('train_progress_div.html', st=st)
#                        
#                    if st.update['result']: # update result
#                        st.update['result'] = False
#                        main_loss_graph(st.epoch, st.main_output['output']['train_loss'], st.main_output['output']['val_loss'])
#                        main_acc_graph(st.epoch, st.main_output['output']['train_acc'], st.main_output['output']['val_acc'])
#                        if st.compare != []:
#                            comparing_graph2(st.main_output, st.compare, 'Loss', 'loss')
#                            comparing_graph2(st.main_output, st.compare, 'Accuracy', 'acc')
##                            comparing_graph(st.main_output, st.compare, 'train_loss', 'loss')
##                            comparing_graph(st.main_output, st.compare, 'val_loss', 'loss')
##                            comparing_graph(st.main_output, st.compare, 'train_acc', 'accuracy')
##                            comparing_graph(st.main_output, st.compare, 'val_acc', 'accuracy')
#                        
#                    if st.update['complete']:
#                        st.update['complete'] = False
#                        response['progress_html'] = render_template('train_progress_div.html', st=st)
#                        del globals()['st']
#                        torch.cuda.empty_cache()
#                        
#                    return jsonify(response)
#                else:
#                    return ''
#                
#            else: # for initializing request
#                if st.update['init']:
#                    return 'done'
#                else:
#                    return ''
        
def exec_test_stream(st):
    st.setup_testing()
    st.test_net()
    st.update['init'] = False
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
            base_meta['title'] = 'Stream Evaluation'
            base_meta['header'] = 'Stream Evaluation'
            global form
            form = TestStreamForm(device_info['gpu_names'])
            # start the stream training thread
            if form.validate_on_submit():
                #Thread(target = (lambda: st.init(form_to_dict(form)))).start()
                st = StreamTrainer(form_to_dict(form))
                print('test_stream/form_success')
                # retrieve training state graph
                #p = torch.load(form.full_model.data, map_location=lambda storage, loc: storage)
                        
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
                base_meta['title'] = 'Stream Evaluation'
                base_meta['header'] = 'Stream Evaluation - '+st.args['output_name']
                p_args = {x : st.args[x] for x in ['modality', 'dataset', 'split', 'network', 'epoch', 'pretrain_model', 'freeze_point', 'base_lr', 
                          'batch_size', 'momentum', 'l2decay', 'dropout', 'lr_scheduler', 'clip_len', 'crop_h', 'crop_w', 
                          'is_mean_sub', 'is_rand_flip']}
                return render_template('test_stream_state.html', base_meta=base_meta, st=st, p_args = p_args, 
                                       no_val=BASE_CONFIG['dataset'][st.args['dataset']]['val_txt'] == [])
            else:
                return render_template('initializing.html', url=url_for('test_stream'))
            
    else: # there is something in the request json
        if 'model' in req.keys(): # update properties and forms default value
            if req['model'] == '':
                return jsonify({'html':'<span style="color: red;">No data available</span>'})
            else:
                p = torch.load(req['model'], map_location=lambda storage, loc: storage)
                p_args = {x : p['args'][x] for x in ['modality', 'dataset', 'split', 'network', 'epoch', 'pretrain_model', 'freeze_point', 'base_lr', 
                          'batch_size', 'momentum', 'l2decay', 'dropout', 'lr_scheduler', 'clip_len', 'crop_h', 'crop_w', 
                          'is_mean_sub', 'is_rand_flip']}
                p_training = {x:p['output'][x] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}
                p_epoch = p['epoch']
                p_args['epoch'] = p_epoch
                return jsonify({'html':render_template('resume_stream_properties.html', args = p_args, training = p_training, epoch = p_epoch), 
                                'device': p['args']['device'], 'clip_len':p['args']['clip_len'], 'debug':p['args']['is_debug_mode']})
    
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
        base_meta['title'] = 'Stream Benchmarking'
        base_meta['header'] = 'Stream Benchmarking'
        global form
        form = InspectStreamForm()
        
        if form.validate_on_submit():
            print('inspect_stream/form_success')
            base_meta['title'] = 'Stream Benchmarking'
            base_meta['header'] = 'Stream Benchmarking - ' + form.main_model.data.split('\\')[-1].split('.')[0]
                        
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
            
            # read in list of class label
            f_in = open(path.join(BASE_CONFIG['dataset'][base_dataset]['base_path'], BASE_CONFIG['dataset'][base_dataset]['label_index_txt']))
            label_list = [x.split(' ')[1] for x in (f_in.read().split('\n'))[:-1]]
            f_in.close()
            
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
            best_test_pkg = {}
            best_acc = {k : 0 for k in train_pkg.keys()}
            
            for name, pkg in test_pkg.items():
                model_method = name.split('=/=')[1]
                model_name = name.split('=/=')[0]
                for method in test_method:
                    if method == model_method:
                        for k, v in pkg['acc'].items():
                            pkg['acc'][k] = round(v * 100, 2)
                        output[model_name]['test_result'][method] = pkg['acc']
                        if pkg['acc']['top-1'] > best_acc[model_name]:
                            best_test_pkg[model_name] = pkg
                            best_acc[model_name] = pkg['acc']['top-1']
                            
            nan_array = np.full((int(BASE_CONFIG['dataset'][base_dataset]['label_num']),int(BASE_CONFIG['dataset'][base_dataset]['label_num'])), np.nan)
                            
            best_test_pkg.update({ x : {'result' : {'score' : nan_array, 'pred' : nan_array}} for x in peer_model[1:] if x not in best_test_pkg.keys()})
            
            main_test = True
            if main_model in best_test_pkg.keys():
                Thread(target=plotly_confusion_matrix, args=(best_test_pkg[main_model]['cfm'], label_list, main_model)).start()
            else:
                main_test = False
                
            if not path.exists('output/benchmark/'+main_model):
                os.makedirs('output/benchmark/'+main_model)
                        
            # training state
            print('inspect_stream/PLOTTING TRAINING STATS')
            main_loss_graph(train_pkg[main_model]['epoch'], train_pkg[main_model]['output']['train_loss'], train_pkg[main_model]['output']['val_loss'], title=main_model, 
                            static_path = 'static/benchmark/', save_path='output/benchmark/'+main_model+'/')
            main_acc_graph(train_pkg[main_model]['epoch'], train_pkg[main_model]['output']['train_acc'], train_pkg[main_model]['output']['val_acc'], title=main_model, 
                           static_path = 'static/benchmark/', save_path='output/benchmark/'+main_model+'/')
            if len(peer_model) > 1: # if there is any comparison
                comparing_graph2(train_pkg[main_model], [model for k, model in train_pkg.items() if k != main_model], 'Loss', 'loss', main_model+'*', 
                                 static_path = 'static/benchmark/', save_path = 'output/benchmark/'+main_model+'/')
                comparing_graph2(train_pkg[main_model], [model for k, model in train_pkg.items() if k != main_model], 'Accuracy', 'acc', main_model+'*', 
                                 static_path = 'static/benchmark/', save_path = 'output/benchmark/'+main_model+'/')
#                comparing_graph(train_pkg[main_model], [model for k, model in train_pkg.items() if k != main_model], 'train_loss', 'loss', main_model+'*')
#                comparing_graph(train_pkg[main_model], [model for k, model in train_pkg.items() if k != main_model], 'val_loss', 'loss', main_model+'*')
#                comparing_graph(train_pkg[main_model], [model for k, model in train_pkg.items() if k != main_model], 'train_acc', 'accuracy', main_model+'*')
#                comparing_graph(train_pkg[main_model], [model for k, model in train_pkg.items() if k != main_model], 'val_acc', 'accuracy', main_model+'*')
            
            # validation plot
            print('inspect_stream/PLOTTING VALIDATION/TESTING STATS')
            for metric in ['pred', 'score']:
                for val_sort in ['desc', 'asc', 'none']:
                    cv_confusion_matrix('static/'+base_dataset+'/'+base_split+'/val/', train_pkg[main_model]['output']['val_result'],
                                        label_list, target=metric, sort=val_sort, 
                                        static_path='static/benchmark/val', save_path='output/benchmark/'+main_model+'/val')
                    cv_confusion_matrix_with_peers('static/'+base_dataset+'/'+base_split+'/val/', 
                                                           train_pkg[main_model]['output']['val_result'], 
                                                           [train_pkg[x]['output']['val_result'] for x in peer_model[1:]], 
                                                           label_list, '*'+main_model, peer_model[1:], 
                                                           target=metric, sort=val_sort, by_peer='none', 
                                                           static_path='static/benchmark/val', save_path='output/benchmark/'+main_model+'/val')
                    if main_model in best_test_pkg.keys():
                        cv_confusion_matrix('static/'+base_dataset+'/'+base_split+'/test/', best_test_pkg[main_model]['result'],
                                            label_list, target=metric, sort=val_sort, 
                                            static_path='static/benchmark/test', save_path='output/benchmark/'+main_model+'/test')
                        cv_confusion_matrix_with_peers('static/'+base_dataset+'/'+base_split+'/test/', 
                                                           best_test_pkg[main_model]['result'], 
                                                           [best_test_pkg[x]['result'] for x in peer_model[1:]], 
                                                           label_list, '*'+main_model, peer_model[1:], 
                                                           target=metric, sort=val_sort, by_peer='none', 
                                                           static_path='static/benchmark/test', save_path='output/benchmark/'+main_model+'/test')
                    for sort_model in peer_model:
                        if sort_model != main_model:
                            cv_confusion_matrix_with_peers('static/'+base_dataset+'/'+base_split+'/val/', 
                                                           train_pkg[main_model]['output']['val_result'], 
                                                           [train_pkg[x]['output']['val_result'] for x in peer_model[1:]], 
                                                           label_list, main_model, peer_model[1:], 
                                                           target=metric, sort=val_sort, by_peer=sort_model, 
                                                           static_path='static/benchmark/val', save_path='output/benchmark/'+main_model+'/val')
                            if main_model in best_test_pkg.keys():
                                cv_confusion_matrix_with_peers('static/'+base_dataset+'/'+base_split+'/test/', 
                                                           best_test_pkg[main_model]['result'], 
                                                           [best_test_pkg[x]['result'] for x in peer_model[1:]], 
                                                           label_list, main_model, peer_model[1:], 
                                                           target=metric, sort=val_sort, by_peer=sort_model, 
                                                           static_path='static/benchmark/test', save_path='output/benchmark/'+main_model+'/test')
            # argument filters
            print('inspect_stream/MATCHING ARGUMENTS')
            args_filter = ['modality', 'dataset', 'split', 'network', 'pretrain_model', 'freeze_point', 'base_lr', 
                           'batch_size', 'momentum', 'l2decay', 'dropout', 'lr_scheduler', 'lr_reduce_ratio', 
                           'step_size', 'last_step', 'patience', 'loss_threshold', 'min_lr', 'clip_len', 
                           'is_mean_sub', 'is_rand_flip']
            for name, pkg in train_pkg.items():
                output[name]['args'] = {arg : train_pkg[name]['args'][arg] for arg in args_filter}
                output[name]['args']['epoch'] = pkg['epoch']
            
            print('inspect_stream/READY')
            return render_template('inspect_stream.html', base_meta=base_meta, output=output, base_model=main_model, dataset=base_dataset, split=base_split, main_test=main_test, 
                                   save_path = 'output/benchmark/'+main_model+'/')
        else:
            for fieldName, errorMessages in form.errors.items():
                for err in errorMessages:
                    print('inspect_stream/form_failed',fieldName, err)
        
        return render_template('inspect_stream_setup.html', base_meta=base_meta, form=form)
    
    else: # ajax request
        if 'base_model' in req.keys():
            base_dataset = req['base_model'].split('\\')[2]
            base_split = req['base_model'].split('\\')[3]
            form.model_compare.choices = [(x,'-'.join([x.split('.')[0].split('\\')[i] for i in [2, 3, 5]]))
                for x in glob('output\\stream\\'+base_dataset+'\\'+base_split+'\\training\\*.pth.tar', recursive=True)]
            return jsonify({'html':form.model_compare})
        
        else:
            return ''
        
@app.route('/visualize_stream', methods=['GET', 'POST'])
def visualize_stream():
    req = request.json
    print('visualize_stream/request.json', req)
    
    if (req == {} or req == None):
        
        if 'sv' not in globals():
            global sv
            sv = None
        else:
            if sv != None:
                sv.is_ready = False
        
        base_meta['title'] = 'Stream Visualization'
        base_meta['header'] = 'Stream Visualization'
        global form
        form = VisualizeStreamForm1(device_info['gpu_names'])
        
        if form.validate_on_submit():
            print('visualize_outer/form_success')
            sv = StreamVis(form_to_dict(form))
            base_meta['title'] = 'Stream Visualization'
            base_meta['header'] = 'Stream Visualization - ' + form.vis_model.data.split('\\')[-1].split('.')[0]
            return redirect(url_for('visualize_panel'))

        else:
            for fieldName, errorMessages in form.errors.items():
                for err in errorMessages:
                    print('visualize_outer/form_failed',fieldName, err)
                    
                    
        return render_template('visualize_stream_setup.html', base_meta = base_meta, form = form)
        
    else:
        return ''

@app.route('/visualize_panel', methods=['GET', 'POST'])
def visualize_panel():
    req = request.json
    print('visualize_panel/request.json', req)
    
    if (req == {} or req == None):
        if 'sv' not in globals():
            return index()
        
        else:
            if sv == None:
                return index()
            else:
                form = VisualizeStreamForm2(sv.args['clip_len'], 
                                            path.join(sv.dataset_path, BASE_CONFIG['dataset'][sv.args['dataset']]['label_index_txt']),
                                            sv.model.convlayer)
                
                if form.validate_on_submit():
                    sv.register_args(form_to_dict(form), form.target_class.choices)
                    # read input first
                    sv.fetch_input()
                    output_name = sv.args['output_name'] + '-' + sv.form_args['clip_name'] + '-' + sv.label_list[int(sv.form_args['target_class'])] + '.jpg'
                    return render_template('visualize_stream_panel.html', base_meta = base_meta, form = form, args=sv.args, 
                                           ready=True, clip_len=form.clip_len.data, save_path='output/vis/'+output_name)
                
                else:
                    sv.is_ready = False
                    for fieldName, errorMessages in form.errors.items():
                        for err in errorMessages:
                            print('visualize_outer/form_failed',fieldName, err)    
                
                return render_template('visualize_stream_panel.html', base_meta = base_meta, form = form, args=sv.args, 
                                       ready=False, clip_len=0, save_path='')
            
    else:
        if 'update' in req.keys():
            if sv.args['modality'] == 'rgb':
                sv.run_visualize_rgb()
            else:
                sv.run_visualize_flow()
            sv.is_ready = False
            return jsonify({'result': True})
        
        else:
            return ''