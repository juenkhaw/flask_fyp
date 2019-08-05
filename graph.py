import plotly.offline as py
import plotly.graph_objs as go

import cv2
import numpy as np
from math import ceil

from glob import glob
from os import path, makedirs

config = {'scrollZoom' : False, 'displayModeBar' : False}
margin = 30
right_margin = 0
legend_font_size = 10
legend_x = 1
legend_y = 1.02

def main_loss_graph(epoch, train_loss, val_loss):
    
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
                   'yaxis':{'title':{'text':'loss'}}
            }}
    py.plot(figure, filename='static/graph/main_loss.html', config=config, auto_open = False)
    
def main_acc_graph(epoch, train_acc, val_acc):
    
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
                   'yaxis':{'title':{'text':'accuracy'}}
            }}
    py.plot(figure, filename='static/graph/main_acc.html', config=config, auto_open = False)
    
def comparing_graph(current, data, title, yaxis):
    
    current_pkg = [go.Scatter(
                x = list(range(1, current['epoch'] + 1)),
                y = current['output'][title],
                name = 'Current',
                showlegend=False
            )]
    compare_pkg = [
                go.Scatter(
                    x = list(range(1, pkg['epoch'] + 1)), 
                    y = pkg['output'][title], 
                    name = pkg['args']['output_name'],
                    showlegend = False
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
    
# blue green red
colors = [(255, 165, 109), (109, 255, 140), (119, 119, 255)]

def cv_confusion_matrix(frame_path, cfm, label, target='pred', sort='none', output_path=''):
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
    
    out_w = padding * ncol + padding + ncol * item_w
    out_h = padding * nrow + padding + item_h * nrow
    
    output = np.ones((out_h, out_w, 3), np.uint8) * 255
    output[:,:,:] = 220
    
    # extract top 5 labels recognized by the network
    top_label = np.argsort(cfm[target], axis = 1)[:, ::-1][:, :4]
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
            y = padding * r + padding + item_h * r
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
            if sorted_ind[index] not in top_label[index]:
                cv2.rectangle(item, 
                              (img_pad * 2 + img_w, 
                               padding + img_pad + bar_pad * 4 + bar_h * 4),
                              (img_pad * 2 + img_w + int(true_value[sorted_ind[index]] / max_value * bar_w), 
                               padding + img_pad + bar_pad * 4 + bar_h * (4 + 1)),
                              colors[0], cv2.FILLED)
                cv2.putText(item, label[sorted_ind[index]], 
                            (img_pad * 2 + img_w, 
                             padding * 2 + img_pad + bar_pad * 4 + bar_h * 4), 
                             cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                cv2.putText(item, str(round(true_value[sorted_ind[index]] * 100, 2)), 
                            (item_w - 35, 
                             padding * 2 + img_pad + bar_pad * 4 + bar_h * 4),
                             cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
                
            np.copyto(output[y : y + item_h, x : x + item_w, :], item)
    
    if not path.exists(output_path):
        makedirs(output_path)
        
    cv2.imwrite(path.join(output_path, target + '_' + sort + '.jpg'), output)
    #cv2.waitKey()
    #cv2.destroyAllWindows()