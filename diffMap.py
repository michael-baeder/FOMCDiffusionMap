# -*- coding: utf-8 -*-
"""
    Utilities for constructing a diffusion map according to various
    implementations.
"""

import numpy as np
import pydiffmap
import matplotlib.pyplot as plt

def cosine_similarity(a,b):
    """
        Calculate the cosine distance between two vectors
    """
    return np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))

def angular_distance(a,b):
    """
        Calculate the angular distance between two data points. Unlike
        cosine similarity (or distance), this is a metric with d(x,x)=0.
    """
    return 2*np.arccos(cosine_similarity(a,b))/np.pi

def euclidean_distance(a,b):
    """
        Calculate the Euclidean distance between two vectors.
    """
    d = lambda u,v: np.dot(u,v)
    return np.sqrt(d(a,a)+d(b,b)-2*d(a,b))

def gaussian_kernel(x, epsilon):
    """
        Calculate the Gaussian kernel measure for a given quantity.
    """
    return np.exp(-np.power(x,2)/epsilon)

def calc_kernel_matrix(X, metric, kernel):
    """
        Given a data matrix, a metric function, and a kernel shape, calculate
        the kernel matrix for a set of data.
    """
    n_items = np.size(X,0)
    n_features = np.size(X,1)
    
    # Initialize the matrix
    D = np.zeros([n_items, n_items])+-np.inf
    
    # Populate half
    for i in range(n_items):
        for j in range(i,n_items):
            D[i,j] = kernel(metric(X[i,:],X[j,:]))
    
    # Symmetrize
    D = np.maximum(D,D.transpose())
    
    return D

def kernel_to_transition(K, alpha):
    """
        Given a kernel matrix K and a normalization parameter alpha, convert
        K into a probability matrix.
    """
    D = np.diag(1/np.power(K.sum(axis=0), alpha))
    P = np.dot(D, np.dot(K,D))
    P = P / P.sum(axis=0)
    return P    

def map_P(X, alpha, n_coords, t, epsilon, metric=angular_distance, kernel=gaussian_kernel):
    """
        Calculate the diffusion coordinates for a dataset given the NxK data
        matrix X, the normalization parameter alpha, the number of coordinates
        desired n_coords, the time/scale factor t, and the kernel bandwidth
        parameter epsilon.
        
        This implementation uses the probability transition matrix, reflected
        by the algorithm in Coiffman, Lafon (2008)
    """
    # Compute the transition matrix
    K = calc_kernel_matrix(X, metric, lambda x: kernel(x, epsilon))
    P = kernel_to_transition(K, alpha)
    
    # Calculate the spectrum
    l, v = np.linalg.eig(np.power(P,t))
    
    # Return the diffusion coordinates
    return v[:,1:(n_coords+1)]

def plot_P_spectrum(X, alpha, t, epsilon, metric=angular_distance, kernel=gaussian_kernel, n_comps=25):
    """
        Given the same params as map_P, plot the eigenvalue spectrum of the
        transition matrix.
    """
    # Compute the transition matrix
    K = calc_kernel_matrix(X, metric, lambda x: kernel(x, epsilon))
    P = kernel_to_transition(K, alpha)
    
    # Calculate the spectrum
    l, v = np.linalg.eig(np.power(P,t))
    l = l[1:n_comps+1]
    plt.plot(range(len(l)), sorted(np.abs(l))[-1::-1])
    plt.title('t={}, $\epsilon$={}, $\\alpha$={}'.format(p['t'],p['epsilon'],p['alpha']))
    return l

    
def map_L(X, alpha, t, epsilon, n_coords, metric='cosine'):
    """
        Calculate the diffusion coordinates for a dataset X given the
        normalization parameter alpha, the time/scale parameter t, the
        kernel bandwidth epsilon, the number of coordinates desired n_coords,
        and the metric between points.
        
        Metric can be one of any of the scipy.spatial.distance metrics.
        
        This implementation uses the pydiffmap library, which implements the
        algorithm on the Laplacian operator L.
    """
    # Construct a kernel using the PyDifFMap interface
    kernel = pydiffmap.kernel.Kernel(epsilon=t,bandwidth_type=epsilon,
                                     k=np.size(X,0),metric=metric)
    
    # Construct a diffusion map using the kernel
    d_map = pydiffmap.diffusion_map.DiffusionMap(kernel, alpha, n_coords)
    return d_map.fit_transform(X)

    
def map_SVD(X, alpha, n_coords, epsilon, t, metric=angular_distance, kernel=gaussian_kernel):
    """
        Calculates the diffusion coordinates for a dataset given the data matrix
        X, the normalization parameter alpha, the number of
        coordinates n_coords, and the kernel bandwidth epsilon.
        
        This implementation is based on the singular-value decomposition of
        the transition matrix. It is an attempt to replicate the implementation
        in MATLAB given by Laurens van der Maaten in the Matlab Toolbox for
        Dimensionality Reduction. More info at:
        https://lvdmaaten.github.io/drtoolbox
    """
    # Calculate the kernel matrix
    K = calc_kernel_matrix(X, metric, lambda x: kernel(x, epsilon))
    
    # Normalize
    p = np.asmatrix(np.sum(K,axis=0)).transpose()
    p_norm = np.power(np.dot(p,p.transpose()), alpha)
    K = K / p_norm
    
    p = np.sqrt(np.asmatrix(np.sum(K,axis=0))).transpose()
    p_norm = np.dot(p,p.transpose())
    K = K / p_norm
    
    u,s,v = np.linalg.svd(K)
    u = u / u[:,0]
    return u[1:(n_coords+1)].transpose()