"""
    Download Federal Funds data using the FedFunds functions and save the
    result as a CSV. Stitch together two series:
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

import FedFunds

# Move to a folder where you want to save the data
os.chdir('..\\FederalFunds')

# Get the data
fred = FedFunds.initialize_fred('FRED API Key.txt')
fed_history = FedFunds.stitch_series(fred, 'DFEDTAR', 'DFEDTARL')
fed_changes = FedFunds.change_series(fed_history)

fed_history.to_csv(path_or_buf='FFR_History.csv',index_label='Date',header=['Rate'])
fed_changes.to_csv(path_or_buf='FFR_Changes.csv',index_label='Date',header=['Change'])