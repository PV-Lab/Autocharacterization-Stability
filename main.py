
import cv2
from PIL import Image
from scipy import signal,ndimage
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import openpyxl 
import pickle
import pandas as pd
import os as os
import tiffile as tif
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy.matlib as matlib
import colorfunctions
from colorfunctions import Sample
from colorfunctions import crop_box, RRC, segment_on_dt, water, time_names_0, output_for_labeling, cc_crop
from colorfunctions import CC_RGB, convert_LAB_RGB, color_calibration, Results
import matplotlib.patches as patches 
import compextractor
import matplotlib.animation as animation
from colorfunctions import point_ave,Edge_detect,warp_image,clean_warp
import mainfunctions
import requests
##################################################################################################################
########################################## COLOR CALLIBRATION ####################################################
##################################################################################################################
# Download the Tiff files from osf 
# Note: Batch 1 and 3 are best, batch 2 requires changing segmentation parameters * 
batch_num = 1 
codes = {
    'deg01':'hwugb',
    'deg11': 'bdyzs',
    'x1' : 'd5knw',
    'deg02': 'fwmnc',
    'deg12': 'wdzj5',
    'x2' : 'sjm7u',
    'deg03': '64ekp',
    'deg13': '3zj8e',
    'x3' : 'r3fj2'
}

mainfunctions.download('tif',url=f"https://osf.io/download/{codes[f'deg0{batch_num}']}",filename='MAFA_deg_0.tif')
mainfunctions.download('tif',url=f"https://osf.io/download/{codes[f'deg1{batch_num}']}",filename='MAFA_deg_1.tif')
mainfunctions.download('xrite',url=f"https://osf.io/download/{codes[f'x{batch_num}']}",filename='xrite.png')

# turn png to jpg 
mainfunctions.png_to_jpg('./Images_simp/xrite.png','./Images_simp/xrite.jpg')
# Reads'sample_names' csv file with information about the experiment 
sheet_obj, Sample_ID, Notes, Time_Step, pixel_to_length, start_image, cut, start_comp, num_pics,time,name,img_name = mainfunctions.unwrap_tiff()
img_path, im = mainfunctions.show_cal_image(img_name)

# Crop the samples our of the whole image 
# These xvals and yvals are crop indices for the min and max x value and the min and max y vals respectively
    # Change xvals and yvals to fit the red box around the area to be cropped (should include the glass slide only)
    # Display the Crop Box as a red rectangle over the image
image_crop_params = {
    'xvals': [40,750],
    'yvals': [180,420]
} 
crop_image, crop_image_pil,crop_params = mainfunctions.crop_im(im, image_crop_params, img_path)

# Segment out the droplets 
# Define the maximum and minimum pixel size of droplets
# Any segmented samples outside this sample range will be deleted 
# Note: if batch is 2 set ks to 15*
seg_params = {
    'large_elements_pixels': 1000, # Define the maximum and minimum pixel size of droplets
    'small_elements_pixels': 0,    # Any segmented samples outside this sample range will be deleted
    'ks' : 11, #11,12,13 ks is the kernel value for dilating the image, enlagring ks reduces clustered droplets 
} 
droplet_count = mainfunctions.segment(crop_image,seg_params)
# Clean ( Remove noise) the segmentation results 
clean_params= {
    'k' : 1, # k = erode kernel to remove noise 
    'kd': 3 # kd = dilate kernel to enlarge segmented droplets 
} 
img_erode,crop_image_erode, PIL_crop_image_erode= mainfunctions.clean(clean_params, seg_params, droplet_count,crop_image,crop_image_pil) 
# Crop out each color in the xrite color card 
xrite_crop_params = {
    'xvals': [308,980], # [X lower bound, X upper bound]
    'yvals': [340,190], # [Y Lower bound, Y upper bound] 200,190,185
    'wid_x': 18, # Width of Crop boxes 
    'hei_x': 10, # height of Crop boxes
    'a': 100, #108 spacex
    'b': 45,  #40 spacey
    'c': 0,   # offsetx
    'd': -1   # offsety
}
xc_ll_x,xc_ll_y = mainfunctions.get_xrite(xrite_crop_params)
sample = output_for_labeling(crop_params, crop_image_pil, img_erode,Sample_ID, droplet_count, crop_image_erode, PIL_crop_image_erode, img_name, Notes, pixel_to_length )
# Run Color Calibration 
RGB_result, LAB_result, CC_result, sample_result, WA_result = Results(xc_ll_x,xc_ll_y,xrite_crop_params['wid_x'],xrite_crop_params['hei_x'],sample.crop_params,num_pics,time,name,start_image)
pickle.dump(RGB_result, open('./Results/RGB_results_f','wb'))

##################################################################################################################
############################## Instability Index Calculation #####################################################
##################################################################################################################
# Extract Results of Color Calibration 
RGB_result, Number_of_drops, num_pics, time,name,drops,droplet_count0,drop_IDs,img_erode,crop_params,start_comp,end_comp,sample_ID,already_processed,cv_params,extract_params = mainfunctions.load_pick()
RGB_result = np.floor(RGB_result).astype(int)
# Detect the COrners of the Glass Slide to warp the image for composition extraction 
# For more explanation on there parameters see : https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
# linesP = cv2.HoughLinesP(edged, 1, np.pi / 180, ts, None, srn,stn)
detect_params = {
        'k' : 5, # kernel size for erode and dilating the original image to get rid of image noise  
        'ap_size' : 3, # these variable are for cv2.Canny(imgray, lb, ub,apertureSize=ap_size)
        'lb' : 30,
        'ub' : 100,  
        'c' : 4, # c value for adaptive tresholding
        'block' : 21, # block size for adaptive tresholding
        'ts' : 42, # 42 Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
        'srn' : 100, # was 140, interesting at 100 minimum line length -Minimum length of line. Line segments shorter than this are rejected.
        'stn' : 11, # 11 maxline gap -Maximum allowed gap between line segments to treat them as single line.
        'min_dis' : 50, # Getting rid of points that are too close to each other 
        'ang_min' :60.0, # 60.0min angle between lines to be considered an edge 
        'k_del' : 20,
        'k_edge' : 5,
        'k_edge_er' : 4

}
crop_image, corners,cv_params = mainfunctions.main_edge_detect(img_erode,name,crop_params,cv_params,already_processed,detect_params)
warp,final_warp,deconst_warp,warp_clean2 = mainfunctions.main_warp(corners,crop_image,img_erode,drop_IDs,Number_of_drops)
# Check Image warped properly
drop_index = {
    'warp_id' : 20, # Droplet in the warped image 
    'og_id' : 20 # Droplet in the original Image 
}
mainfunctions.warp_check(drop_index, final_warp, img_erode, drop_IDs)
# Extract the composition of the materials based on the MMD System 
comp_params = {
    'rasterSpeed' : 38, # mm/s gocde raster speed
    'rasterScaleY' :  0.8, # scale gcode raster pattern in Y-dim
    'rasterScaleX' : 0.935, # scale gcode raster pattern in X-dim
    'rasterOffsetY' : 4, # offset gcode raster pattern in X-dim
    'rasterOffsetX' : 15 # offset gcode raster pattern in X-dim
} 
extract_params, order, comp = mainfunctions.main_comp(comp_params,already_processed,final_warp,extract_params)
# Check that the droplets have been properly ordered 
mainfunctions.order_check(Number_of_drops,img_erode,order)
# Extract the Instability Indices 
Ic,Test_array_RGB,c = mainfunctions.extract_ic(RGB_result, Number_of_drops,drops,num_pics)
# Order the Indices based on composition 
Ic_reorder,Test_array_reorder = mainfunctions.order_ic(Ic, Test_array_RGB,Number_of_drops,drop_IDs,order,comp)
# Plot Results 
spacing = {
    'y_int' : 10, # Spacing between the y-axis tick marks 
    'x_int' : 20  # Spacing between the x-axis tick marks
} 
mainfunctions.plot_grand(num_pics,Test_array_RGB,Test_array_reorder, time, start_comp, end_comp, comp, Number_of_drops, spacing)
mainfunctions.degrade_gif(num_pics,RGB_result)
# Plots of Individual Droplets 
drop_IDs = {
    'desi' : [8,20,29,43,54,0]
}
mainfunctions.individual_drops(c,drop_IDs,Ic_reorder,Test_array_reorder)