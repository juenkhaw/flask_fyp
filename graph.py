import plotly.offline as py
import plotly.graph_objs as go

import cv2
import numpy as np
from math import ceil

from glob import glob
from os import path, makedirs
from shutil import copy2

config = {
        'scrollZoom' : False, 
        'displayModeBar' : 'hover'
        }
margin = 30
right_margin = 0
legend_font_size = 10
legend_x = 1
legend_y = 1.02

def main_loss_graph(epoch, train_loss, val_loss, title='', static_path = 'static/training/', save_path=''):
    
    figure = {'data': [
                go.Scatter(
                    x = list(range(1, epoch + 1)), 
                    y = train_loss, 
                    name='train_loss',
                    fill='tozeroy',
                ),
                go.Scatter(
                    x = list(range(1, epoch + 1)), 
                    y = val_loss, 
                    name='val_loss',
                    fill = 'tonexty',
            )], 
            'layout': {
                   'legend':{'x':legend_x, 'y':legend_y}, 
                   'margin':{'l':margin,'r':right_margin,'t':margin,'b':margin}, 
                   'font':{'size':legend_font_size},
                   'xaxis':{'title':{'text':'epoch'}},
                   'yaxis':{'title':{'text':'loss'}},
                   'title':{'text':title}
            }}
    py.plot(figure, filename=static_path+'main_loss.html', config=config, auto_open = False)
    if save_path != '':
        copy2(static_path+'main_loss.html', save_path+'main_loss.html')
        #py.plot(figure, filename=save_path+'main_loss.html', config=config, auto_open = False)
    
def main_acc_graph(epoch, train_acc, val_acc, title='', static_path = 'static/training/', save_path=''):
    
    figure = {'data': [
                go.Scatter(
                    x = list(range(1, epoch + 1)), 
                    y = train_acc, 
                    name='train_acc',
                    fill='tozeroy',
                ),
                go.Scatter(
                    x = list(range(1, epoch + 1)), 
                    y = val_acc, 
                    name='val_acc',
                    fill = 'tonexty',
            )], 
            'layout': {
                   'legend':{'x':legend_x, 'y':legend_y}, 
                   'margin':{'l':margin,'r':right_margin,'t':margin,'b':margin}, 
                   'font':{'size':legend_font_size},
                   'xaxis':{'title':{'text':'epoch'}},
                   'yaxis':{'title':{'text':'accuracy'}}, 
                   'title':{'text':title}
            }}
    py.plot(figure, filename=static_path+'main_acc.html', config=config, auto_open = False)
    if save_path != '':
        copy2(static_path+'main_acc.html', save_path+'main_loss.acc')
        #py.plot(figure, filename=save_path+'main_acc.html', config=config, auto_open = False)
    
def comparing_graph(current, data, title, yaxis, main_name = 'Current'):
    
    current_pkg = [go.Scatter(
                x = list(range(1, current['epoch'] + 1)),
                y = current['output'][title],
                name = main_name,
                showlegend=True
            )]
    compare_pkg = [
                go.Scatter(
                    x = list(range(1, pkg['epoch'] + 1)), 
                    y = pkg['output'][title], 
                    name = pkg['args']['output_name'],
                    showlegend = True,
                    hoverlabel = {'namelength' : 30}
                ) for pkg in data
            ]
    current_pkg.extend(compare_pkg)
    
    figure = {'data': current_pkg, 
            'layout': {
                    'legend':{'x':legend_x, 'y':legend_y}, 
                    'margin':{'l':margin,'r':margin,'t':margin,'b':margin}, 
                    'font':{'size':legend_font_size},
                    'title':{'text':title},
                    'xaxis':{'title':{'text':'epoch'}},
                    'yaxis':{'title':{'text':yaxis}}
                    }
            }
    py.plot(figure, filename='static/graph/compare_'+title+'.html', config=config, auto_open = False)
    
def comparing_graph2(current, data, title, yaxis, main_name = 'Current', 
                     static_path = 'static/training/', save_path=''):
    
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd'  # muted purple
    ]
    
    current_pkg = [go.Scatter(
                x = list(range(1, current['epoch'] + 1)),
                y = current['output']['train_'+yaxis],
                name = 'train_' + main_name,
                showlegend=True,
                hoverlabel = {'namelength' : 30},
                line = dict(color=colors[0]),
                mode = 'lines'
            ), go.Scatter(
                x = list(range(1, current['epoch'] + 1)),
                y = current['output']['val_'+yaxis],
                name = 'val_' + main_name,
                showlegend=True,
                hoverlabel = {'namelength' : 30},
                line = dict(color=colors[0], dash='dash'),
                mode = 'lines'
            )]
    
    compare_pkg1 = [
                go.Scatter(
                    x = list(range(1, pkg['epoch'] + 1)), 
                    y = pkg['output']['train_'+yaxis], 
                    name = 'train_' + pkg['args']['output_name'],
                    showlegend = True,
                    hoverlabel = {'namelength' : 30},
                    line = dict(color=colors[i + 1]),
                    mode = 'lines'
                ) for i, pkg in enumerate(data)
            ]
    compare_pkg2 = [
                go.Scatter(
                    x = list(range(1, pkg['epoch'] + 1)), 
                    y = pkg['output']['val_'+yaxis], 
                    name = 'val_' + pkg['args']['output_name'],
                    showlegend = True,
                    hoverlabel = {'namelength' : 30},
                    line = dict(color=colors[i + 1], dash='dash'),
                    mode = 'lines'
                ) for i, pkg in enumerate(data)
            ]
    
    current_pkg.extend(compare_pkg1)
    current_pkg.extend(compare_pkg2)
    
    figure = {'data': current_pkg, 
            'layout': {
                    'legend':{'x':legend_x, 'y':legend_y}, 
                    'margin':{'l':margin,'r':margin,'t':margin,'b':margin}, 
                    'font':{'size':legend_font_size},
                    'title':{'text':title},
                    'xaxis':{'title':{'text':'epoch'}},
                    'yaxis':{'title':{'text':yaxis}}
                    }
            }
    py.plot(figure, filename=static_path + 'compare_'+yaxis+'.html', config=config, auto_open = False)
    if save_path != '':
        copy2(static_path+'compare_'+yaxis+'.html', save_path+'compare_'+yaxis+'.html')
        #py.plot(figure, filename=save_path + 'compare_'+yaxis+'.html', config=config, auto_open = False)
    
# blue green red
colors = [(255, 165, 109), (109, 255, 140), (119, 119, 255)]

def cv_confusion_matrix(frame_path, cfm, label, target='pred', sort='none',
                        static_path = 'static/', save_path=''):
    """
    cfm: confusion matrix output from stream module
    label: label listing from dataloader
    target: display info for [pred/score]
    sort: display the label corresponding to the order of precision [none/asc/desc]
    NOTE: asc = worst to best, desc = best to worst
    """
    sample_frames = np.array(glob(path.join(frame_path, '*.jpg')))
    ncol = 4
    nrow = ceil(cfm[target].shape[0]/ncol)
    
    padding = 10
    item_w = 250
    item_h = 120
    
    img_pad = 5
    img_h = 80
    img_w = 80
    
    bar_h = 16
    bar_w = 150
    bar_pad = 5
    
    remark_h = 20
    
    out_w = padding * ncol + padding + ncol * item_w
    out_h = padding * nrow + padding + item_h * nrow + remark_h
    
    output = np.ones((out_h, out_w, 3), np.uint8) * 255
    output[:,:,:] = 220
    
    cv2.rectangle(output, (padding+105,0), (padding+200, 15), colors[1], cv2.FILLED)
    cv2.rectangle(output, (padding+205,0), (padding+385, 15), colors[0], cv2.FILLED)
    cv2.rectangle(output, (padding+400,0), (padding+480, 15), colors[2], cv2.FILLED)
    cv2.putText(output, 'Label indicator: Correct label' + ' Correct label behind top-5 ' + ' Wrong label', 
                (padding, padding), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,0))
    
    # extract top 5 labels recognized by the network
    top_label = np.argsort(cfm[target], axis = 1)[:, ::-1][:, :5]
    top_value = np.array([cfm[target][i, ind] for i, ind in enumerate(top_label)])
    max_value = np.max(top_value)
    sorted_ind = list(range(cfm[target].shape[0]))
    true_value = [cfm[target][i, i] for i in range(cfm[target].shape[0])]
    
    if sort != 'none':
        sorted_ind = np.argsort(true_value)
        if sort == 'desc':
            sorted_ind = sorted_ind[::-1]
        sample_frames = sample_frames[sorted_ind]
        #label = np.array(label)[sorted_ind]
        top_label = top_label[sorted_ind]
        top_value = top_value[sorted_ind]
    
    for r in range(0, nrow):
        for c in range(0, ncol):
            x = padding * c + padding + item_w * c
            y = padding * r + padding + item_h * r + remark_h
            index = r * ncol + c
            if index >= cfm[target].shape[0]: 
                break

            # empty item figure
            item = np.ones((item_h, item_w, 3), np.uint8) * 255
            
            # title            
            cv2.putText(item, str(sorted_ind[index] + 1) + ' ' + label[sorted_ind[index]], 
                        (0, padding), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0))
            # sample rgb frame
            frame = cv2.imread(sample_frames[index])
            frame = cv2.resize(frame, (img_h, img_w))
            np.copyto(item[padding + img_pad : padding + img_pad + img_h, img_pad : img_pad + img_w], frame)
            #np.copyto(output[y + padding + img_pad : y + padding + img_pad + img_h, x + img_pad : x + img_pad + img_w], frame)
            
            if all([np.isnan(x) for x in top_value[index]]): 
                cv2.putText(item, 'N/A', 
                                (img_pad * 2 + img_w + 30, 
                                 item_h//2), 
                                 cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255))
            
            else:
            
                for b in range(4):
                    if top_value[index, b] == 0:
                        break
                    cv2.rectangle(item, 
                                  (img_pad * 2 + img_w, 
                                   padding + img_pad + bar_pad * b + bar_h * b),
                                  (img_pad * 2 + img_w + int(top_value[index, b] / max_value * bar_w), 
                                   padding + img_pad + bar_pad * b + bar_h * (b + 1)),
                                  colors[1 if sorted_ind[index] == top_label[index, b] else 2], cv2.FILLED)
                    cv2.putText(item, label[top_label[index, b]], 
                                (img_pad * 2 + img_w, 
                                 padding * 2 + img_pad + bar_pad * b + bar_h * b), 
                                 cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                    cv2.putText(item, str(round(top_value[index, b] * 100, 2)), 
                                (item_w - 35, 
                                 padding * 2 + img_pad + bar_pad * b + bar_h * b),
                                 cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                                
                # display true label
                if sorted_ind[index] not in top_label[index] or sorted_ind[index] == top_label[index][4]:
                    cv2.rectangle(item, 
                                  (img_pad * 2 + img_w, 
                                   padding + img_pad + bar_pad * 4 + bar_h * 4),
                                  (img_pad * 2 + img_w + int(true_value[sorted_ind[index]] / max_value * bar_w), 
                                   padding + img_pad + bar_pad * 4 + bar_h * (4 + 1)),
                                  colors[1 if sorted_ind[index] == top_label[index, 4] else 0], cv2.FILLED)
                    cv2.putText(item, label[sorted_ind[index]], 
                                (img_pad * 2 + img_w, 
                                 padding * 2 + img_pad + bar_pad * 4 + bar_h * 4), 
                                 cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                    cv2.putText(item, str(round(true_value[sorted_ind[index]] * 100, 2)), 
                                (item_w - 35, 
                                 padding * 2 + img_pad + bar_pad * 4 + bar_h * 4),
                                 cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                    
            np.copyto(output[y : y + item_h, x : x + item_w, :], item)
        
    cv2.imwrite(path.join(static_path, target + '_' + sort + '.jpg'), output)
    if save_path != '':
        if not path.exists(save_path):
            makedirs(save_path)
        copy2(path.join(static_path, target + '_' + sort + '.jpg'), path.join(save_path, target + '_' + sort + '.jpg'))
        #cv2.imwrite(path.join(save_path, target + '_' + sort + '.jpg'), output)
    #cv2.imwrite('output.jpg', output)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    
peer_colors = [(3, 186, 252), (120, 248, 255), (186, 252, 3), (242, 131, 229)]
    
def cv_confusion_matrix_with_peers(frame_path, cfm, peer_cfm, label, base_name, peer_names, 
                                   target='pred', sort='none', by_peer='none',
                                   static_path = 'static/', save_path=''):
    
    sample_frames = np.array(glob(path.join(frame_path, '*.jpg')))
    ncol = 4
    nrow = ceil(cfm[target].shape[0]/ncol)
    
    padding = 10
    item_w = 250
    item_h = 120
    
    remark_h = 30
    
    img_pad = 5
    img_h = 80
    img_w = 80
    
    bar_h = 16
    bar_w = 150
    half_bar = int(bar_w/2)
    bar_pad = 5
    
    out_w = padding * ncol + padding + ncol * item_w
    out_h = padding * nrow + padding + item_h * nrow + remark_h
    
    output = np.ones((out_h, out_w, 3), np.uint8) * 255
    output[:,:,:] = 220
    
    cv2.rectangle(output, (padding+105,0), (padding+145, 15), colors[1], cv2.FILLED)
    cv2.rectangle(output, (padding+160,0), (padding+245, 15), colors[0], cv2.FILLED)
    cv2.rectangle(output, (padding+255,0), (padding+375, 15), colors[2], cv2.FILLED)
    cv2.putText(output, 'Rank indicator: Top-1 ' + ' Within top-5 ' + ' Fall behind top-5', 
                (padding, padding), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,0))
    
    peer_names_str = ', '.join(peer_names)
    cv2.putText(output, 'Peer Order: ' + base_name + '*, ' + peer_names_str, 
                (padding, padding + 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,0))
    
    # extract top 5 labels recognized by the network
    top_label = np.argsort(cfm[target], axis = 1)[:, ::-1][:, :5] # top 5 label with highest score (n, 5)
    top_value = np.array([cfm[target][i, ind] for i, ind in enumerate(top_label)]) # highest score of top 5 label (n, 5)
    max_value = np.max(top_value) # max score amongst all labels (1)
    sorted_ind = list(range(cfm[target].shape[0])) # sorted index of labels (n)
    true_value = [cfm[target][i, i] for i in range(cfm[target].shape[0])] # score of the true label (n)
    
    # extract performance diff between base and each peers
    peer_top_label = []
    peer_diffs = []
    max_diffs = []
    for pcfm in peer_cfm:
        # extract score of true label in peer cfm
        peer_top_label.append(np.argsort(pcfm[target], axis = 1)[:, ::-1][:, :5])
        pcfm_true_value = [pcfm[target][i, i] for i in range(pcfm[target].shape[0])]
        if len(true_value) > len(pcfm_true_value): # if peer was in peek debug mode while base wasnt
            pcfm_true_value.extend([np.nan for x in range(len(true_value) - len(pcfm_true_value))])
        else: # else if the base was in peek debug mode while peer wasnt
            pcfm_true_value = pcfm_true_value[:len(true_value)]
        temp_diff = np.array(pcfm_true_value) - true_value
        peer_diffs.append(temp_diff)
        max_diffs.append(max([np.nanmax(temp_diff), abs(np.nanmin(temp_diff))]))
        
    if sort != 'none':
        if by_peer == 'none':
            sorted_ind = np.argsort(true_value)
            if sort == 'desc':
                sorted_ind = sorted_ind[::-1]
            sample_frames = sample_frames[sorted_ind]
            top_label = top_label[sorted_ind]
        else:
            peer_index = peer_names.index(by_peer)
            sorted_ind = np.argsort(peer_diffs[peer_index])
            if sort == 'desc':
                sorted_ind = sorted_ind[::-1]
                sorted_ind = np.concatenate((sorted_ind[-len(peer_cfm[peer_index][target]):], sorted_ind[:-len(peer_cfm[peer_index][target])]))
            sample_frames = sample_frames[sorted_ind]
            top_label = top_label[sorted_ind]
        
    #print(peer_top_label)
    
    for r in range(0, nrow):
        for c in range(0, ncol):
            x = padding * c + padding + item_w * c
            y = padding * r + padding + item_h * r + remark_h
            index = r * ncol + c
            if index >= cfm[target].shape[0]: 
                break

            # empty item figure
            item = np.ones((item_h, item_w, 3), np.uint8) * 255
            
            # title            
            cv2.putText(item, str(sorted_ind[index] + 1) + ' ' + label[sorted_ind[index]], 
                        (0, padding), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0))
            
            # sample rgb frame
            frame = cv2.imread(sample_frames[index])
            frame = cv2.resize(frame, (img_h, img_w))
            np.copyto(item[padding + img_pad : padding + img_pad + img_h, img_pad : img_pad + img_w], frame)
            
            if all([np.isnan(x) for x in top_value[index]]): 
                cv2.putText(item, 'N/A', 
                                (img_pad * 2 + img_w + 30, 
                                 item_h//2), 
                                 cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255))
            else: 
                # base performance
                cv2.rectangle(item, 
                              (img_pad * 2 + img_w, 
                               padding + img_pad),
                              (img_pad * 2 + img_w + int(true_value[sorted_ind[index]] / max_value * bar_w), 
                               padding + img_pad + bar_h),
                              colors[1 if sorted_ind[index] == top_label[index, 0] else 2 if sorted_ind[index] not in top_label[index] else 0], cv2.FILLED)
                cv2.putText(item, (base_name if len(base_name) <= 15 else base_name[:15]), 
                            (img_pad * 2 + img_w, 
                             padding * 2 + img_pad), 
                             cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                cv2.putText(item, str(round(true_value[sorted_ind[index]] * 100, 2)), 
                            (item_w - 35, 
                             padding * 2 + img_pad),
                             cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                            
                # differences with peers
                for b in range(0, len(peer_diffs)):
                    c = b + 1
                    if np.isnan(peer_diffs[b][sorted_ind[index]]):
                        cv2.putText(item, peer_names[b], 
                                    (img_pad * 2 + img_w, 
                                     padding * 2 + img_pad + bar_pad * c + bar_h * c), 
                                     cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                        cv2.putText(item, 'N/A', 
                                    (item_w - 40, 
                                     padding * 2 + img_pad + bar_pad * c + bar_h * c),
                                     cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255))
                    else:
                        if peer_diffs[b][sorted_ind[index]] >= 0:
                            cv2.rectangle(item, 
                                          (img_pad * 2 + img_w + half_bar, 
                                           padding + img_pad + bar_pad * c + bar_h * c),
                                          (img_pad * 2 + img_w + int(peer_diffs[b][sorted_ind[index]] / max_diffs[b] * half_bar + half_bar), 
                                           padding + img_pad + bar_pad * c + bar_h * (c + 1)),
                                          colors[1 if sorted_ind[index] == peer_top_label[b][sorted_ind[index], 0] else 2 if sorted_ind[index] not in peer_top_label[b][sorted_ind[index]] else 0], cv2.FILLED)
                        else:
                            cv2.rectangle(item, 
                                          (img_pad * 2 + img_w + half_bar - abs(int(peer_diffs[b][sorted_ind[index]] / max_diffs[b] * half_bar)), 
                                           padding + img_pad + bar_pad * c + bar_h * c),
                                          (img_pad * 2 + img_w + half_bar, 
                                           padding + img_pad + bar_pad * c + bar_h * (c + 1)),
                                          colors[1 if sorted_ind[index] == peer_top_label[b][sorted_ind[index], 0] else 2 if sorted_ind[index] not in peer_top_label[b][sorted_ind[index]] else 0], cv2.FILLED)
                        cv2.putText(item, ('*' if peer_names[b] == by_peer else '') + (peer_names[b] if len(peer_names[b]) <= 15 else peer_names[b][:15]), 
                                    (img_pad * 2 + img_w, 
                                     padding * 2 + img_pad + bar_pad * c + bar_h * c), 
                                     cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                        cv2.putText(item, ('+' if peer_diffs[b][sorted_ind[index]] >= 0 else '') + str(round(peer_diffs[b][sorted_ind[index]] * 100, 2)), 
                                    (item_w - 43, 
                                     padding * 2 + img_pad + bar_pad * c + bar_h * c),
                                     cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                
                # seperator line
                cv2.line(item, (img_pad * 2 + img_w + half_bar, padding + img_pad + bar_pad + bar_h), 
                         (img_pad * 2 + img_w + half_bar, padding + img_pad + bar_pad * 5 + bar_h * 5 - padding), 
                         (0, 0, 0))
            
            np.copyto(output[y : y + item_h, x : x + item_w, :], item)
            
    cv2.imwrite(path.join(static_path, target + '_' + sort + '_' + by_peer + '.jpg'), output)
    if save_path != '':
        if not path.exists(save_path):
            makedirs(save_path)
        copy2(path.join(static_path, target + '_' + sort + '_' + by_peer + '.jpg'), path.join(save_path, target + '_' + sort + '_' + by_peer + '.jpg'))
        #cv2.imwrite(path.join(save_path, target + '_' + sort + '_' + by_peer + '.jpg'), output)
    
def plotly_confusion_matrix(cfm, label, title):
    print('inspect_stream/PREPARING CFM')
    cfm2 = np.copy(cfm)
    for i in range(cfm2.shape[0]):
        cfm2[i, i] = -1
    data = [go.Heatmap(
            colorscale = 'Reds',
            x = label[:cfm.shape[1]],
            y = label[:cfm.shape[0]],
            z = [list(pred) for pred in cfm2]
            )]
    layout = {
            'title': {'text': title},
            'xaxis': {'title': {'text':'Predicted label'}},
            'yaxis': {'title': {'text':'Actual label'}},
            'autosize': False,
            'width': 1000 if cfm.shape[1] <= 20 else 50 * cfm.shape[1], 
            'height': 1000 if cfm.shape[0] <= 20 else 50 * cfm.shape[0],
            'annotations': [{'x':label[x], 
                             'y':label[y], 
                             'font':{'color':'white'}, 
                             'text':str(int(cfm[y, x])), 
                             'xref':'x1',
                             'yref':'y1',
                             'showarrow': False} for y in range(cfm.shape[1]) for x in range(cfm.shape[0])]
            }
    fig = go.Figure(data=data, layout=layout)
    #fig.write_image('C:/Users/Juen/Desktop/Gabumon/Blackhole/UTAR/Subjects/FYP/flask_fyp/output.png')
    print('inspect_stream/EXPORTING CFM')
    if not path.exists('output/benchmark/'+title):
        makedirs('output/benchmark/'+title)
    py.plot(fig, filename='static/benchmark/cfm.html', config=config, auto_open=False)
    if not path.exists('output/benchmark/'+title):
        makedirs('output/benchmark/'+title)
    copy2('static/benchmark/cfm.html', 'output/benchmark/'+title+'cfm.html')
    #py.plot(fig, filename='output/benchmark/'+title+'cfm.html', config=config, auto_open=False)
    print('inspect_stream/CFM DONE')
    