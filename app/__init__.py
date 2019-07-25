# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:19:15 2019

@author: Juen
"""

from flask import Flask

import os

# setting static folder
app = Flask(__name__, static_folder = 'static')
# avoid browser from caching static contents (updating graphs with new contents)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'abc123'

root_dir = os.path.dirname(os.getcwd())

from app import routes