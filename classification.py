# -*- coding: utf-8 -*-
"""
    Collection of utilities for using the Diffusion Map for classification
    problems.
"""

import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

def labels_from_csv(file_name, doc_dates, change_tol=0, window_size=7):
    """
        Load a time series from the \\LabelSets directory and parse it into
        a set of labels for the data.
    """
    data_series = pd.read_csv('..\\LabelSets\\' + file_name +'.csv')
    data_series['Date'] = [datetime.datetime.strptime(t,'%Y-%m-%d') for t in data_series['Date']]
    data_series = data_series.set_index('Date')
    return series_to_label(data_series, doc_dates)

def load_labels(file_name, doc_dates):
    """
        Utility for getting labels from a document that already has categorical
        information.
    """
    data_series = pd.read_csv('..\\LabelSets\\'+file_name+'.csv')
    data_series['Date'] = [datetime.datetime.strptime(t,'%Y-%m-%d') for t in data_series['Date']]
    data_series = data_series.set_index('Date')
    
    date_grid = pd.DataFrame(data=None, index=doc_dates)
    
    label_series = date_grid.align(data_series,join='outer',method='pad')[1]
    return [label_series.loc[d].values[0] for d in doc_dates]
    
    
def series_to_label(time_series, doc_dates, change_tol=0, window_size=7):
    """ 
        Utility for converting a numeric time series to changes centered on
        a set of dates. Returns a sequence of labels for each document
        date as "Up", "Down", or "Unchanged".
    """
    date_grid = pd.DataFrame(data=None, index=doc_dates)
    time_series = time_series.align(date_grid,join='outer',method='pad')[0]
    labels = []
    for dt in doc_dates:
        dtIdx = time_series.index.get_loc(dt)
        idxChange = [max(dtIdx-window_size,0), min(dtIdx+window_size,len(time_series)-1)]
        tsChange = time_series.iloc[idxChange,0].diff().values[-1]
        if tsChange > change_tol:
            labels+=['Up']
        elif tsChange < -change_tol:
            labels+=['Down']
        else:
            labels+=['Unchanged']
    return labels

def dict_to_label(dict_data, doc_dates):
    """
        Given a dictionary of labels, match them to a set of document dates
        based on the last value available.
    """
    labels = []
    dict_dates = [datetime.datetime.strptime(d,'%Y%m%d') for d in dict_data.keys()]
    for dt in doc_dates:
        last_date = max([d for d in dict_dates if dt > d])       
        labels+=[dict_data[datetime.datetime.strftime(last_date,'%Y%m%d')]]
    return labels

def classify_absorb(Y, labels):
    """
        Given a set of diffusion coordinates and labels for data, create a
        classifier for each datapoint. Labels which are labels as "Censored"
        will be treated as unknown and based on weighted average distance,
        while points with other labels will be treated as known.
        
        Since the diffusion distance is the probability-weighted path length
        between two points, and is the euclidean distance in Y, we take the
        probability of absorption as the weighted average of the distances
        to the closest member of each class.
    """
    all_labels = list(set(labels))
    real_labels = [l for l in all_labels if l!='Censored']
    p = np.zeros([np.size(Y,0),len(real_labels)])
    for i in range(Y.shape[0]):
        if labels[i]!='Censored':
            p[i,:] = [1 if labels[i]==real_labels[j] else 0 for j in range(len(real_labels))]
        else:
            min_dists = np.array([min([((Y[i,:]-Y[k,:])**2).sum()
                for k in range(Y.shape[0]) if labels[k]==l])
                    for l in real_labels])
            p[i,:] = list(min_dists/min_dists.sum())
    return p, real_labels

def calc_cross_entropy(q, real_labels, labels):
    """
        Given a classification scheme q and the set of actual labels, compute
        the cross-entropy and then average. Since the real probability
        is always one to the label, the calculation is simplified.
    """
    ce = 0
    for i in range(q.shape[0]):
        cei = np.asscalar(-np.log(q[i,[j for j in range(len(real_labels)) if labels[i]==real_labels[j]]]))
        ce+= cei
    return ce

def test_accuracy(Y, labels, k_classify=30, n_runs=100):
    """
        Test the accuracy of the classifier by running numerous trials with
        data randomly censored.
    """
    res = np.zeros(n_runs)
    for i in range(n_runs):
        censor_idx = np.random.choice(range(len(labels)), k_classify, replace=False)
        censored_labels = ['Censored' if i in censor_idx else labels[i] for i in range(len(labels))]
        q, lbls = classify_absorb(Y, censored_labels)
        res[i] = calc_cross_entropy(q, lbls, labels)
    return res

def plot_labels(Y, labels):
    """
        Plots a set of diffusion coordinates, with the colors indicated by
        labels.
    """
    unique_labels = list(set(labels))
    if Y.shape[1]>2:
        fig, axs = plt.subplots(*list(np.repeat(Y.shape[1],2)))
        plot_handles = []
        for i in range(Y.shape[1]):
            for j in range(Y.shape[1]):
                if i!=j:
                    colors = cycle(['r', 'g', 'k', 'c', 'm', 'y', 'b'])
                    plot_handles = []
                    for l in unique_labels:
                        c = next(colors)
                        idx = [i for i in range(len(labels)) if labels[i]==l]
                        x = np.array(Y[idx,i])
                        y = np.array(Y[idx,j])
                        ph = axs[i,j].scatter(x,y,c=c,label=l)
                        plot_handles+=[ph]
                else:
                    if len(plot_handles):
                        axs[i,j].legend(handles=plot_handles)
    else:
        plot_handles = []
        colors = cycle(['r', 'g', 'k', 'c', 'm', 'y', 'b'])
        for l in unique_labels:
            c = next(colors)
            idx = [i for i in range(len(labels)) if labels[i]==l]
            x = np.array(Y[idx,0])
            y = np.array(Y[idx,1])
            ph = plt.scatter(x,y,c=c,label=l)
            plot_handles+=[ph]
        plt.legend(handles=plot_handles)
        
def param_sweep(x_params, x_name, y_params, y_name, base_params, func, labels, test_params={'n_runs':10},verbose=True):
    """
        Sweep two sets of parameters and construct a heatmap showing how
        the accuracy changes.
    """
    X,Y = np.meshgrid(np.array(x_params), np.array(y_params))
    meanRes = np.zeros(X.shape)
    stdRes = np.zeros(X.shape)
    
    p = base_params
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if verbose:
                print(str(i)+'/'+str(j))
            p[x_name] = X[i,j]
            p[y_name] = Y[i,j]
            if test_params is not None:
                res = test_accuracy(func(p),labels,**test_params)
            else:    
                res = test_accuracy(func(p),labels)
            meanRes[i,j] =res.mean()
            stdRes[i,j] = res.std()
            
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, meanRes, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(format_param(x_name))
    ax.set_ylabel(format_param(y_name))
    ax.set_zlabel('$\mu(H)$')
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, stdRes, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(format_param(x_name))
    ax.set_ylabel(format_param(y_name))
    ax.set_zlabel('$\sigma(H)$')
    
    
def format_param(name):
    formats = {'n_coords':'$n_{cooords}$', 
               't':'$t$',
               'epsilon':'$\\epsilon$',
               'alpha':'$\\alpha$'}
    return formats[name]
            