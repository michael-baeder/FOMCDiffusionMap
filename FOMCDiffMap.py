# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:07:48 2020

@author: Michael
"""

# Import external libaries
import os, datetime
from pydiffmap import diffusion_map as dm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Move to the folder that contains all of the project code
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Load project code
import TextProcessing as tp
import utils
import diffMap
import classification

# Construct the data matrix of TF-IDF scores
corpus = tp.make_corpus(utils.list_contents('..\\FOMCStatements'))
tfidf = tp.calc_tf_idf(corpus)

# Set-up the diffusion map parameters
X = np.vstack(list(tfidf.values()))
base_params = {'X':X,'alpha':1,'epsilon':0.25,'n_coords':2,'t':0.75}
func = lambda x: diffMap.map_P(**x)

# Set-up different labels
doc_dates = [datetime.datetime.strptime(k,'%Y%m%d') for k in tfidf.keys()]
all_labels = dict()
for l in utils.list_contents('..\\LabelSets','csv'):
    l = l.split('\\')[-1].split('.csv')[0]
    l_name = re.compile('History').sub('',l)
    if l_name in ['Presidents','Chairs']:
        l = classification.load_labels(l,doc_dates)
    else:
         l = classification.labels_from_csv(l,doc_dates)
    all_labels[l_name] = l
    
# Exploratory analysis
# By president
p = base_params
p['n_coords'] = 2
p['t'] = 0.25
p['epsilon'] = 0.025
l = 'Presidents'
classification.plot_labels(func(p), all_labels[l])
fig = plt.gcf()
fig.set_size_inches(5, 5)
fig.suptitle('{}, t={}, $\epsilon$={}, $\\alpha$={}'.format(l,p['t'],p['epsilon'],p['alpha']))

# By president
p = base_params
p['n_coords'] = 10
p['t'] = 0.8
p['epsilon'] = 0.025
l = 'SP500'
classification.plot_labels(func(p), all_labels[l])
fig = plt.gcf()
fig.set_size_inches(5, 5)
fig.suptitle('{}, t={}, $\epsilon$={}, $\\alpha$={}'.format(l,p['t'],p['epsilon'],p['alpha']))


# FFR
p = base_params
p['n_coords'] = 4
p['t'] = 0.1
p['epsilon'] = 0.025
l = 'FFR'
classification.plot_labels(func(p), all_labels[l])
fig = plt.gcf()
fig.set_size_inches(10, 10)
fig.suptitle('{}, t={}, $\epsilon$={}, $\\alpha$={}'.format(l,p['t'],p['epsilon'],p['alpha']))

# Surface plots
classification.param_sweep(range(2,10), 'n_coords', np.linspace(0,1,20), 't', base_params, func, all_labels['FFR'])

# Comparison




