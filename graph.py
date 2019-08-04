import plotly.offline as py
import plotly.graph_objs as go

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