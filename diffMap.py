# -*- coding: utf-8 -*-
"""
Utilities for my own home-grown diffusion map.
"""

import numpy as np

def get_coordinates(D,t,n):
    """
        Get the diffusion coordinates for a dataset as modeled by a diffusion
        matrix D.
    """
    [V,W] = np.linalg.eig(np.power(D,t))
    W = np.dot(W, np.repeat(np.matrix(V), np.size(W,1), axis=0))
    return W[:,:n]