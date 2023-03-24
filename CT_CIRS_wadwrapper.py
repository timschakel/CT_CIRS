#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:25:48 2022

@author: tschakel
"""

__version__ = '20221017'
__author__ = 'tschakel'

# runfile('/smb/user/tschakel/BLD_RT_RESEARCH_DATA/USER/tschakel/projects/wadqc/QAtests/CT_CIRS/CT_CIRS_wadwrapper.py', args='-r results.json -c config/petct_cirs_config.json -d /smb/user/tschakel/BLD_RT_RESEARCH_DATA/USER/tschakel/projects/wadqc/QAtests/CT_CIRS/data/dataset1', wdir='/smb/user/tschakel/BLD_RT_RESEARCH_DATA/USER/tschakel/projects/wadqc/QAtests/CT_CIRS')


# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput

import matplotlib
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import CT_CIRS_lib

if __name__ == "__main__":
    data, results, config = pyWADinput()
    
    # Log which series are found
    data_series = data.getAllSeries()
    print("The following series are found:")
    for item in data_series:
        print(item[0]["SeriesDescription"].value+" with "+str(len(item))+" instances")
        
        
    """
    Perform the analysis of the CT CIRS Phantom Test.
    """
    
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            CT_CIRS_lib.acqdatetime(data, results, action)
        elif name == 'analysis':
            CT_CIRS_lib.analysis(data, results, action)

    results.write()
