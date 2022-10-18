#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:42:38 2022

@author: tschakel
"""
from wad_qc.modulelibs import wadwrapper_lib
from wad_qc.module.moduledata import getValue,isFiltered,applyFilters
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import Circle


def acqdatetime(data, results, action):
    """
    Get the date and time of acquisition
    """
    filters = action["filters"]
    datetime_series = data.getInstanceByTags(filters["datetime_filter"])
    dt = wadwrapper_lib.acqdatetime_series(datetime_series[0])
    results.addDateTime('AcquisitionDateTime', dt) 

def point_in_circle(point, circle):
    return ((point[0]-circle.center[0])**2 + (point[1]-circle.center[1])**2) < circle.radius**2

def get_vals_circle_ROI(image, circle):
    values = []
    for y in range(circle.center[1]-np.int0(circle.radius), circle.center[1]+np.int0(circle.radius)):
        for x in range(circle.center[0]-np.int0(circle.radius), circle.center[0]+np.int0(circle.radius)):
            if point_in_circle((x, y), circle):
                values.append(image[y,x])
    
    return values
    
def analysis(data, results, action):
    #apparently CT data does not need to be transposed?
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0],headers_only=False,do_transpose=False)
    
    acqdate = dcmInfile.info.StudyDate
    acqtime = dcmInfile.info.StudyTime
    
    # Number of slices varies, depending on operator
    # First mask the phantom
    ct_masked = np.zeros_like(pixeldataIn)
    ct_masked[ (pixeldataIn > -200) ] = 1

    # Fill holes
    # ct_masked = ndimage.binary_fill_holes(ct_masked,structure=np.ones((10,10,10)))
    
    # Find center of phantom
    com = np.int0(np.round(ndimage.center_of_mass(ct_masked)))
    
    # Tube locations
    # Offsets in pixels wrt center of mass
    # Assumes robust center of mass finding and no changes in acquisition resolution
    # (could switch to mm instead of voxels to account for resolution)
    tubes = np.array([[0,0,25],         # 0: water_1e000
                      [0,-170,35],      # 1: air_0e000
                      [120,-120,35],    # 2: lung_exh_0e489
                      [170,0,35],       # 3: bone_1e695
                      [120,120,35],     # 4: adipose_0e949
                      [0,170,35],       # 5: bone_1e456 (should be bone_1e695?)
                      [-120,120,35],    # 6: lung_inh_0e190
                      [-170,0,35],      # 7: muscle_1e043
                      [-120,-120,35],   # 8: bone_1e117 (not used currently?)
                      [0,-328,35],      # 9: o_muscle_1e043
                      [231,-231,35],    # 10: o_bone_1e456
                      [328,0,35],       # 11: o_lung_inh_0e190
                      [231,231,35],     # 12: o_breast_0e976
                      [0,315,35],       # 13: o_liver_1e052
                      [-231,231,35],    # 14: o_bone_1e117
                      [-328,0,35],      # 15: o_lung_exh_0e489
                      [-231,-231,35],   # 16: o_adipose_0e949
                      [-375,-375,35]])  # 17: o_air_0e000
    
    # Image to plot: center slice with the ROIs
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(pixeldataIn[com[0],:,:],cmap='gray')
    ax.scatter(com[2],com[1])
    ax.axis('off')
    title = "CIRS phantom measurement "+acqdate+" "+acqtime
    ax.set_title(title,fontsize=15)
    filename = "CIRS phantom ROIs.jpg"
    
    # Take 5 slices to get statistics from (-2 and +2 of central slice)
    hu_medians=[]
    hu_stdevs=[]
    for (x,y,r) in tubes:
        hu_values=[]
        for slice in range(com[0]-2,com[0]+2):
            circ = Circle((com[2] + x, com[1] + y) , radius = r, fill = False,ec='r')
            circ_vals = get_vals_circle_ROI(pixeldataIn[slice,:,:],circ)
            hu_values.append(circ_vals)
            if slice == com[0]:
                ax.add_patch(circ)
        
        hu_medians.append(np.median(hu_values))
        hu_stdevs.append(np.std(hu_values))
        
    # Combine results
    # Sort according to legacy results:
    fig.savefig(filename,dpi=100)
    results.addObject("CIRS phantom ROIs", filename)

    results.addFloat("HU median AIR", np.int0(np.round(np.mean([hu_medians[1],hu_medians[17]]))))
    results.addFloat("HU median LUNG INHALE", np.int0(np.round(np.mean([hu_medians[6],hu_medians[11]]))))
    results.addFloat("HU median LUNG EXHALE", np.int0(np.round(np.mean([hu_medians[2],hu_medians[15]]))))
    results.addFloat("HU median ADIPOSE", np.int0(np.round(np.mean([hu_medians[4],hu_medians[16]]))))
    results.addFloat("HU median BREAST", np.int0(np.round(hu_medians[12])))
    results.addFloat("HU median WATER", np.int0(np.round(hu_medians[0])))
    results.addFloat("HU median MUSCLE", np.int0(np.round(np.mean([hu_medians[7],hu_medians[9]]))))
    results.addFloat("HU median LIVER", np.int0(np.round(hu_medians[13])))
    results.addFloat("HU median BONE 1.117", np.int0(np.round(hu_medians[14])))
    results.addFloat("HU median BONE 1.456", np.int0(np.round(hu_medians[10])))
    results.addFloat("HU median BONE 1.695", np.int0(np.round(np.mean([hu_medians[5],hu_medians[3]]))))
    
    results.addFloat("HU stdev AIR", np.int0(np.round(np.mean([hu_stdevs[1],hu_stdevs[17]]))))
    results.addFloat("HU stdev LUNG INHALE", np.int0(np.round(np.mean([hu_stdevs[6],hu_stdevs[11]]))))
    results.addFloat("HU stdev LUNG EXHALE", np.int0(np.round(np.mean([hu_stdevs[2],hu_stdevs[15]]))))
    results.addFloat("HU stdev ADIPOSE", np.int0(np.round(np.mean([hu_stdevs[4],hu_stdevs[16]]))))
    results.addFloat("HU stdev BREAST", np.int0(np.round(hu_stdevs[12])))
    results.addFloat("HU stdev WATER", np.int0(np.round(hu_stdevs[0])))
    results.addFloat("HU stdev MUSCLE", np.int0(np.round(np.mean([hu_stdevs[7],hu_stdevs[9]]))))
    results.addFloat("HU stdev LIVER", np.int0(np.round(hu_stdevs[13])))
    results.addFloat("HU stdev BONE 1.117", np.int0(np.round(hu_stdevs[14])))
    results.addFloat("HU stdev BONE 1.456", np.int0(np.round(hu_stdevs[10])))
    results.addFloat("HU stdev BONE 1.695", np.int0(np.round(np.mean([hu_stdevs[5],hu_stdevs[3]]))))