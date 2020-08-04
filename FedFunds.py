# -*- coding: utf-8 -*-
"""
Utilities for getting historical Federal Funds data.

An API key is needed in order to query FRED. You can get one after making an
account at:

    https://research.stlouisfed.org/useraccount/apikeys
"""

import pandas as pd
from fredapi import Fred
from utils import throttle

def initialize_fred(file):
    """ 
        Initialize a FRED API session with an API key stored in a text file.
    """
    with open(file) as f:
        fred = Fred(api_key = f.read())
    return fred

@throttle(milliseconds=500)
def req_series(fred,series):
    """
        Wrapper for querying FRED API to space out queries.
    """
    return fred.get_series(series)
    
    
def stitch_series(fred,*argv):
    """
        Download and join multiple series along the time dimension.
    """
    return pd.concat([req_series(fred,name) for name in argv])

def change_series(data):
    """
        Filter a pandas series to only the times when there is change.
    """
    data = data.diff()
    data = data[data!=0]
    return data.iloc[1:]