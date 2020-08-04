# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:07:48 2020

@author: Michael
"""

import os, datetime
from pydiffmap import diffusion_map as dm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Move to the folder that contains all of the project code
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import TextProcessing as tp
import utils
import diffMap

corpus = tp.make_corpus(utils.list_contents('..\\FOMCStatements'))
tfidf = tp.calc_tf_idf(corpus)
D = tp.calc_cosine_similarity(tfidf)

ffr_hist = pd.read_csv('..\\FederalFunds\\FFR_History.csv')
ffr_hist['Date'] = [datetime.datetime.strptime(t,'%Y-%m-%d') for t in ffr_hist['Date']]
ffr_hist = ffr_hist.set_index('Date')

# Use the FFR_changes series to label statements where there was action
doc_order = list(tfidf.keys())
doc_labels = []
for do in doc_order:
    dt = datetime.datetime.strptime(do,'%Y%m%d')
    dtIdx = ffr_hist.index.get_loc(dt)
    idxChange = [max(dtIdx-7,0), min(dtIdx+7,len(ffr_hist)-1)]
    ffrChange = ffr_hist.iloc[idxChange,0].diff().values[-1]
    rgb = 'black'
    if ffrChange>0:
        rgb='red'
    elif ffrChange<0:
        rgb='green'
    doc_labels = doc_labels + [rgb]
    

tList = [[0.05, 0.25], [0.5, 5]]

fig, axs = plt.subplots(*list(np.shape(tList)))
for i in range(len(tList)):
    for j in range(len(tList)):
        t = tList[i][j]
        [V,W] = np.linalg.eig(np.power(D,t))
        x = np.array(W[:,0])
        y = np.array(W[:,1])
        #colors = np.matrix(doc_labels)
        axs[i,j].scatter(x, y, c = doc_labels)
        axs[i,j].set_title(str(t))
        
t = 1
n_coordinates = 5
fig, axs = plt.subplots(n_coordinates, n_coordinates)
[V,W] = np.linalg.eig(np.power(D,t))
for i in range(n_coordinates):
    for j in range(n_coordinates):
        if i != j:
            x = np.array(W[:,i])
            y = np.array(W[:,j])
            axs[i,j].scatter(x,y,c=doc_labels)
            axs[i,j].set_title(str(i) + ' vs. ' + str(j))
        else:
            axs[i,j].hist(np.array(W[:,i]))
            
