"""
    Download data from the ST Louis Fed database using the FRED functions and
    save the result as a CSV.
    
    The Fed funs series is formed by stitching together two series:
        DFEDTAR : The target rate. Prior to 2008, this was targeted
                  specifically by the Fed.
        DFEDTARL: After 2008, the Fed moved to targeting a range in the Federal
                  Funds rate. This is the more widely followed lower bound.
"""

import os
import pandas as pd

# Move to the folder that contains all of the project code
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import FredUtils

# Move to a folder where you want to save the data
os.chdir('..\\LabelSets')

# Get the data
fred = FredUtils.initialize_fred('FRED API Key.txt')
fed_history = FredUtils.stitch_series(fred, 'DFEDTAR', 'DFEDTARL')

fed_history.to_csv(path_or_buf='FFRHistory.csv',index_label='Date',header=['Rate'])

twoYearHistory = FedFunds.req_series(fred,'DGS2')
twoYearHistory.to_csv(path_or_buf='2YHistory.csv', 
                   index_label='Date',header=['Rate'])

tenYearHistory = FedFunds.req_series(fred,'DGS10')
tenYearHistory.to_csv(path_or_buf='10YHistory.csv', 
                   index_label='Date',header=['Rate'])

slope = twoYearHistory.align(tenYearHistory,join='inner',method='Backfill')
slope = slope[1] - slope[0]
slope.to_csv(path_or_buf='SlopeHistory.csv',
             index_label='Date',header=['Rate'])