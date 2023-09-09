
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



def output_for_labeling(crop_params, crop_img_pil, imgn_erode, sample_ID, water,crop_image_erode,
                        PIL_crop_image_erode, img_name, Notes, PTL):
    # Create a Sample class
    # Create Drop_IDs List 
    num_drops = np.unique(imgn_erode)
    if np.size(np.where(num_drops==0)) != 0:
        num_drops = np.extract(num_drops!=0, num_drops)
    drop_IDs = num_drops 
    Number_of_drops = np.size(drop_IDs)
    # Create the 3D array drops where each layer is zero everywhere except the location of a single droplet
    drops_rows = np.size(imgn_erode, 0)
    drops_col = np.size(imgn_erode, 1)
    drops = np.zeros((Number_of_drops,drops_rows,drops_col))
    for i in range(Number_of_drops):
        n = drop_IDs[i]
        drop_loc = np.copy(imgn_erode)
        drop_loc[drop_loc!=n] = 0
        drops[i,:,:] = drop_loc
    sample = Sample(crop_params, crop_img_pil,sample_ID, drop_IDs, drops, Number_of_drops, water, imgn_erode,
                    crop_image_erode, PIL_crop_image_erode, img_name, Notes, PTL)
    # Save all information in a pickle format
    sample.save()
    return sample



def time_names_0(time_step,cut,start_image):
    # function that takes in Tiff file and impacts the images and labels them according to the time stamp
    # cut = 30
    i = 0 
    # Read each Tiff Image store in img_raw and count the number of pictures and store as i 
    for name in os.listdir('./Images_Tiff'):
        #print(name)
        i+=1
        #print(f'./Images_Tiff/{name}')
        #print(os.path.isfile(f'./Images_Tiff/{name}'))
        if os.path.isfile(f'./Images_Tiff/{name}'):
            if i == 1:
                #print(i)
                img_raw = tif.imread(f'./Images_Tiff/{name}')
            else :
                # print(i)
                img_raw = np.concatenate((img_raw,tif.imread(f'./Images_Tiff/{name}')))
    # img_raw_start = img_raw[start_image-1, :,:]
    #print(img_raw.shape)
    img_raw_start = np.copy(img_raw[start_image:,:,:,:])
    num_pics = np.size(img_raw_start,0)
    num = math.floor(num_pics/cut)
    time = np.arange(1,(num)+1,1)*(time_step*cut)
    time.reshape(num,1)
    strings = [str(x) for x in time]
    name0 = ['Img_']*num
    name1 = [''.join(element) for element in zip(name0 ,strings)]
    jpg = ['.jpg'] * num
    name = [''.join(element) for element in zip(name1 ,jpg)]
    
    # Save the tiff images as jpg with the given names 
    for i in range(num):
        j = cut*i
        if (j>num_pics-1):
            break 
        im = Image.fromarray(img_raw_start[j,:,:])
        im.save(f'./Images_simp/{name[i]}')
    #print(img_raw_start.shape)
    
    return num, time , name

def crop_box(im,xvals,yvals,verbose):
    # Redraw the box to confirm it what they want
    # Determine the Rectangle lower left point
    ll_y = math.ceil(min(yvals))
    ll_x = math.ceil(min(xvals))
    # Determine rectangle width
    width = math.ceil(max(xvals)) - ll_x
    # Determine rectangle height
    height = math.ceil(max(yvals)) - ll_y
    fig, ax = plt.subplots(1, figsize=(1248 / 200, 1024 / 300))
    ax.imshow(im)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    # Add rectangle
    ax.add_patch(Rectangle((ll_x, ll_y), width, height, edgecolor='red', fill=False))
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.title(f"Cropped Area Sample")
    if verbose:
        plt.show()
    plt.close()
    return ll_x, ll_y, width, height

def RRC(img_path, rotate_crop_params,package):
    '''
    Rotates and crops the given image.

    Inputs:
    img                  := image path
    rotate_crop_params   := dictionary of values: {theta, x1, x2, y1, y2}, where
        theta            := angle of counter clockwise rotation
        x1               := start pixel of x-axis crop
        x2               := end pixel of x-axis crop
        y1               := start pixel of y-axis crop
        y2               := end pixel of y-axis crop

    Ouputs:
    img                  := rotated and cropped image
    '''
    if package == 'cv2':
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # read images 
    elif package == 'pil':
        img = Image.open(img_path, 'r')
    rotated = ndimage.rotate(img, rotate_crop_params['theta'])  # reads image and rotates
    img = rotated[rotate_crop_params['y1']:rotate_crop_params['y2'],
          rotate_crop_params['x1']:rotate_crop_params['x2']]  # crops image
    return img

def segment_on_dt(a, img, threshold):
    '''
    Implements watershed segmentation.

    Inputs:
    a         := the raw image input
    img       := threshold binned image
    threshold := RGB threshold value

    Outputs:
    lbl       := Borders of segmented droplets
    wat       := Segmented droplets via watershed
    lab       := Indexes of each segmented droplet
    '''
    # estimate the borders of droplets based on known and unknown background + foreground (computed using dilated and erode)
    border = cv2.dilate(img, None, iterations=1)
    border = border - cv2.erode(border, None)
    # segment droplets via distance mapping and thresholding
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, threshold, 255, cv2.THRESH_BINARY)
    # obtain the map of segmented droplets with corresponding indices
    lbl, ncc = ndimage.label(dt)
    lbl = lbl * (255 / (ncc + 1))
    lab = lbl
    # Completing the markers now.
    lbl[border == 255] = 255
    lbl = lbl.astype(np.int32)
    a = cv2.cvtColor(a,
                     cv2.COLOR_GRAY2BGR)  # we must convert grayscale to BGR because watershed only accepts 3-channel inputs
    wat = cv2.watershed(a, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl, wat, lab  # return lab, the segmented and indexed droplets

def water(image, small_elements_pixels, large_elements_pixels,k):
    '''
    Applies watershed image segmentation to separate droplet pixels from background pixels.

    Inputs:
    image                   := input droplet image to segment
    large_elements_pixels   := Cleans large elements that contain more than specified number of pixels.


    Outputs:
    droplet_count           := Image of droplet interiors indexed by droplet number
    binarized               := Binary image indicating total droplet area vs. empty tube space
    ''' 
    RGB_threshold = 0
    pixel_threshold = 0
    # Added these Lines 
    
    kernel = np.ones((k,k), np.uint8)
    image= cv2.dilate(image, kernel)
    # added these lines 
    img = image.copy()
    img = 255 - img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 3)
    _, img_bin = cv2.threshold(img, 0, 255,
                               # threshold image using Otsu's binarization 
                               # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
                               cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                               np.ones((4, 4), dtype=int))
    # first fold of watershed to remove white centers
    result, water, labs = segment_on_dt(a=img, img=img_bin,
                                        threshold=RGB_threshold)  
    # segment droplets from background and return indexed droplets


    # remove small/large elements
    uniq_full, uniq_counts = np.unique(water,
                                       return_counts=True)  # get all unique watershed indices with pixel counts
    large_elements = uniq_full[uniq_counts > large_elements_pixels]  # mask large elements based on number of pixels
    small_elements = uniq_full[uniq_counts < small_elements_pixels] # mask small elements based on number of pixels
    for n in range(len(large_elements)):
        water[water == large_elements[n]] = 0  # remove all large elements
    for n in range(len(small_elements)):
        water[water == small_elements[n]] = 0  # remove all small elements

    result[result == 255] = 0
    droplet_count = result.copy()
    return water

def cc_crop(x_vals,y_vals, x_space,y_space,img,spacex,spacey,offsetx,offsety,verbose):
    # x_vals is a 1x2 array with min and max x value of the whole color card crop box 
    # y_vals is the same 
    # x_space is spacing in the x -direction between boxes 
    # y_vals is spacing in the y direction between boxes 

    r = 4
    c = 6
    xvals = np.zeros((r,c))
    yvals = xvals
    fig, ax = plt.subplots(1,figsize=(1248 / 200, 1024 / 300))
    ax.imshow(img)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    width = x_space
    height = y_space
    ll_x_a = []
    ll_y_a = []
    for i in range(c):
        for j in range(r):
            ll_y = math.ceil(min(y_vals)+ (j)*(y_space + spacey)+ offsety*i)
            ll_x = math.ceil(min(x_vals)+(i)*(x_space + spacex)+offsetx*j)
            ax.add_patch(Rectangle((ll_x, ll_y), width, height, edgecolor='red', fill=False))
            ll_x_a.append(ll_x)
            ll_y_a.append(ll_y)
    plt.title(f'Cropped Colors of Xrite Color Card')
    if verbose:
        plt.show()
    plt.close()
    return ll_x_a, ll_y_a

# Function to extract Color Card RGB data from 1 image  
def CC_RGB(cc_ll_x, cc_ll_y, width,height,img_path):
    # set Crop indices 
    R_drop_cc = []
    G_drop_cc=[]
    B_drop_cc = []
    R_hi_drop_cc = []
    G_hi_drop_cc = []
    B_hi_drop_cc = []
    R_lo_drop_cc = []
    G_lo_drop_cc = []
    B_lo_drop_cc = []
    for j in range(24):
        crop_params = {
        'theta': 0,
        'x1': cc_ll_x[j],
        'x2': cc_ll_x[j]+width,
        'y1': cc_ll_y[j], 
        'y2': cc_ll_y[j]+height
        }
        crop_image = RRC(img_path, crop_params,'pil')
        R_channel = np.array(crop_image, dtype=np.uint8)[:,:,0]
        G_channel = np.array(crop_image, dtype=np.uint8)[:,:,1]
        B_channel = np.array(crop_image, dtype=np.uint8)[:,:,2]
        R_drop_cc.append([round(np.mean(R_channel))])
        G_drop_cc.append([round(np.mean(G_channel))])
        B_drop_cc.append([round(np.mean(B_channel))])
        R_hi_drop_cc.append([np.percentile(np.sort(R_channel, axis = None), 95)])
        G_hi_drop_cc.append([np.percentile(np.sort(G_channel, axis = None), 95)])
        B_hi_drop_cc.append([np.percentile(np.sort(B_channel, axis = None), 95)])
        R_lo_drop_cc.append([np.percentile(np.sort(R_channel, axis = None), 5)])
        G_lo_drop_cc.append([np.percentile(np.sort(G_channel, axis = None), 5)])
        B_lo_drop_cc.append([np.percentile(np.sort(B_channel, axis = None), 5)])
        # For each square create RGB series over time, the matrices returned contain nested array where R_drop[0] for 
        # example has all values over time of Red channel of the first Color Card color 
    
    return R_drop_cc,G_drop_cc,B_drop_cc, R_hi_drop_cc,G_hi_drop_cc,B_hi_drop_cc, R_lo_drop_cc,G_lo_drop_cc,B_lo_drop_cc
 
def convert_LAB_RGB(data, to_space, from_space):
    # Input:
    # - data: a np array with dimensions (n_samples, {optional
    #   dimension: n_times}, n_color_coordinates=3) (e.g., a direct output of
    #   'rgb_extractor()' or 'rgb_extractor_Xrite_CC()')
    # - from_space: choose either 'RGB' or 'Lab'
    # - to_space: choose either 'RGB' or 'Lab'
    # Output:
    # - converted: a np array with the same dimensions than in the input

    n_d = data.ndim
    if n_d == 2:
        data = np.expand_dims(data, 1)
    elif n_d != 3:
        raise Exception('Faulty number of dimensions in the input!')
    if (from_space == 'RGB') and (to_space == 'LAB'):
        # Values from rgb_extractor() are [0,255] so let's normalize.
        data = data/255
        # Transform to color objects (either sRGBColor or LabColor).
        data_objects = np.vectorize(lambda x,y,z: sRGBColor(x,y,z))(
            data[:,:,0], data[:,:,1], data[:,:,2])
        # Target color space
        color_space = matlib.repmat(LabColor, *data_objects.shape)
        # Transform from original space to new space 
        converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
            data_objects, color_space)
        # We got a matrix of color objects. Let's transform to a 3D matrix of floats.
        converted = np.transpose(np.vectorize(lambda x: (x.lab_l, x.lab_a, x.lab_b))(
            converted_objects), (1,2,0))

    elif (from_space == 'LAB') and (to_space == 'RGB'):
        data_objects = np.vectorize(lambda x,y,z: LabColor(x,y,z))(
            data[:,:,0], data[:,:,1], data[:,:,2])
        color_space = matlib.repmat(sRGBColor, *data_objects.shape)
        converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
            data_objects, color_space)
        converted = np.transpose(np.vectorize(lambda x: (x.rgb_r, x.rgb_g, x.rgb_b))(
            converted_objects), (1,2,0))
        # Colormath library interprets rgb in [0,1] and we want [0,255] so let's
        # normalize to [0,255].
        converted = converted*255
    else:
        raise Exception('The given input space conversions have not been implemented.')
    if n_d == 2:
        converted = np.squeeze(converted)
    return converted

# New function for faster color calibration  
# Taken from previous color calibration code  in 2023
def color_calibration(drop,ll_x, ll_y, width,heigth, file,XR,XG,XB, XR_hi,XG_hi,XB_hi,
                      XR_lo,XG_lo,XB_lo,reference_CC_lab):

    # Let's extract the rgb colors from our color passport picture.
    # Convert the color card into LAB values 

    img_array = np.array(np.stack((XR,XG,XB), axis= 2),dtype = np.uint8)
    #RGB_reshaped = img_array.reshape(24,3)
    #CC_lab = convert_LAB_RGB(RGB_reshaped, 'LAB', 'RGB')
    RGB_reshaped = img_array.reshape(24,1,3)
    CC_lab = cv2.cvtColor((RGB_reshaped/255).astype('float32'), cv2.COLOR_RGB2LAB)
    CC_lab = CC_lab.reshape(24,3)
    
    
    # print('This is X_lab')
    # print(CC_lab)
    # Convert droplet RGBs into LAB values 
    #print("Lab")
    #sample_lab = convert_LAB_RGB(drop, 'LAB', 'RGB')
    #sample_lab = cv2.cvtColor(drop, cv2.COLOR_RGB2LAB)
    #drop = cv2.cvtColor(drop, cv2.COLOR_BGR2RGB)
    # drop is BGR since RRC used 'cv2'
    sample_lab = cv2.cvtColor((drop/255).astype('float32'), cv2.COLOR_BGR2LAB)
    # Reorganize the reference to match the colors orders in the color card
    

    
    # X_hi_lab = X_hi_lab[order]
    # X_lo_lab =  X_lo_lab[order]
    # sample_lab = sample_lab[order]
    
    ###########################
    # Color calibration starts.
    #print("Section 1")
    # Number of color patches in the color chart.
    N_patches = 24
    
    # Let's create the weight matrix for color calibration using 3D thin plate
    # spline.

    # Data points of our color chart in the original space.
    # Pi = [1 l_color1 a_color1 b_color_1] 
    # 24 x 4 
    P = np.concatenate((np.ones((N_patches,1)), CC_lab), axis=1)
    # Data points of our color chart in the transformed space.
    # 24x3
    V = reference_CC_lab
    # Shape distortion matrix, K
    # 24x24 
    K = np.zeros((N_patches,N_patches))
    R = np.zeros((N_patches,N_patches))
    # %%timeit -n 1000
    # How to calculate K_IJ w/o a for loop 
    X = P[:,1].reshape((1,N_patches))
    Y = P[:,2].reshape((1,N_patches))
    Z = P[:,3].reshape((1,N_patches))
    X_mat = np.zeros((N_patches,N_patches))
    X_mat[:,:] = np.transpose(X)
    Y_mat = np.zeros((N_patches,N_patches))
    Y_mat[:,:] = np.transpose(Y) 
    Z_mat = np.zeros((N_patches,N_patches))
    Z_mat[:,:] = np.transpose(Z) 
    rx = np.square(X_mat - X)
    ry = np.square(Y_mat - Y)
    rz = np.square(Z_mat - Z)
    r= np.sqrt(rx+ry+rz)
    K= 2* np.multiply(np.square(r),np.log(r+10**(-20)))
    # Linear and non-linear weights WA:
    numerator = np.concatenate((V, np.zeros((4,3))), axis=0)
    denominator = np.concatenate((K,P), axis=1)
    denominator = np.concatenate((denominator,
                                  np.concatenate((np.transpose(P),
                                                  np.zeros((4,4))),axis=1)), axis=0)
    WA = np.matmul(np.linalg.pinv(denominator), numerator)
    #print("check")
    # Checking if went ok. We should get the same result than in V (except for
    # the 4 bottom rows)
    CC_lab_double_transformation = np.matmul(denominator,WA)
    #print('Color chart patches in reference Lab:', reference_CC_lab,
    #      'Color chart patches transformed to color calibrated space and back - this should be the same than above apart
    # from the last 4 rows',
    #      CC_lab_double_transformation, 'subtracted: ', reference_CC_lab-CC_lab_double_transformation[0:-4,:])
    # print('Checking if color transformation is successful - all values here should be near zero:/n', reference_CC_lab
    # CC_lab_double_transformation[0:-4,:])
    #print("Section 2") 
    # Let's perform color calibration for the sample points!
    N_samples = sample_lab.shape[0]
    N_times = sample_lab.shape[1]
    # print(K_test)
    P_new = np.ones((N_samples, N_times,4))
    P_new[:,:,1:4] = np.copy(sample_lab)
    # 190x560x4 
    K_new = np.zeros((N_samples, N_times,N_patches))
    # 190x560x24
    X_n = P_new[:,:,1].reshape((N_samples,1,N_times))
    Y_n = P_new[:,:,2].reshape((N_samples,1,N_times))
    Z_n = P_new[:,:,3].reshape((N_samples,1,N_times)) 
    #190x1x560
    X_mat = np.zeros((N_samples,N_times,N_patches))
    X_mat[:,:] = X_n.reshape((N_samples,N_times,1))
    Y_mat = np.zeros((N_samples,N_times,N_patches))
    Y_mat[:,:] = Y_n.reshape((N_samples,N_times,1)) 
    Z_mat = np.zeros((N_samples,N_times,N_patches))
    Z_mat[:,:] = Z_n.reshape((N_samples,N_times,1)) 
    # (190, 560, 24)
    X_sub = np.zeros((N_samples,1,N_patches))
    X_sub[:,:] = X
    Y_sub = np.zeros((N_samples,1,N_patches))
    Y_sub[:,:] = Y
    Z_sub = np.zeros((N_samples,1,N_patches))
    Z_sub[:,:] = Z
    rx = np.square(X_mat - X_sub)
    ry = np.square(Y_mat - Y_sub)
    rz = np.square(Z_mat - Z_sub)
    # 190,560,24
    r= np.sqrt(rx+ry+rz)
    K_new =2* np.multiply(np.square(r),np.log(r+10**(-20)))
    #190,560,24 
    K_tt = np.copy(K_new)
    P_sub = np.zeros((N_samples,N_patches,4))
    # 190x24x4
    P_sub[:,:] = P
    dennom = np.concatenate((K_new,P_new),axis=2)
    denden = np.concatenate((P_sub.reshape(N_samples,4,N_patches), np.zeros((N_samples,4,4))), axis=2)
    # Try changing P to P_new to see if this fixes the color calibration errors 
    sample_lab_cal = np.matmul(np.concatenate((dennom, denden), axis=1), WA)
    # Remove zeros, i.e., the last four rows from the third dimension.
    sample_lab_cal = sample_lab_cal[:,0:-4,:]
    #print("sample_lab_cal dtype")
    #print(sample_lab_cal.dtype)
    # 190x560x3
    ################################
    # Color calibration is done now.
    
    # Let's transform back to rgb.
    # print("Transform")
    ##sample_rgb_cal = convert_LAB_RGB(sample_lab_cal, 'RGB', 'LAB')
    #print("float 64")
    #print(sample_lab_cal)
    #print(np.rint(sample_lab_cal).astype(np.int8))
    #print("float32")
    #print(sample_lab_cal.astype(np.float32))
    sample_rgb_cal = cv2.cvtColor(sample_lab_cal.astype('float32'), cv2.COLOR_LAB2RGB)
    sample_rgb_cal = sample_rgb_cal*255
    
    #sample_rgb_cal = convert_LAB_RGB(sample_lab_cal, 'RGB', 'N')
    
    # X_hi_lab = convert_LAB_RGB([XR_hi,XG_hi,XB_hi], 'LAB')
    # X_lo_lab = convert_LAB_RGB([XR_lo, XG_lo, XB_lo],'LAB')
    
    
    # Let's return both lab and rgb calibrated values.
    return sample_rgb_cal, sample_lab_cal, CC_lab, sample_lab, WA

def Results(xc_ll_x, xc_ll_y, wid_x, hei_x,crop_params,num_pics,time,names,start_img,verbose=False):
    # This function runs color calibration on the droplets over all files
    # Code structure: For each image extract all calibrated droplet colors  
    # Inputs: 
        # xc_ll_x, xc_ll_y = The crop box lower left and lower right indices for Xrite Colour Checker
        # wid_x, hei_x = The width and height of the crop boxes for Xrite Colour Checker
        # crop params = Crop parameters of the reference image  
        # Time_steps = Time step between each image defined in the sample_names spreadsheet
        # start_img = The index of the starting image in the image time series 
            # (removes the first couple of frames of the timeseries data)
    # Outputs:
        # drops_rgb_cal_x, drops_lab_cal_x = RGB and LAB colour images of the color clibrated images
            # where drops_rgb_cal_x[x][0] is the color clibrated image at time step x from the start image 
        # cut = The number of the images skiped between in your data set in order to reduce processign time. 
            # If cut = 1 you extract the color data of every image 
    
    drops_rgb_cal_x = []
    drops_lab_cal_x = []
    CC_out= []
    sample_out= []
    WA_out= []
    reference_CC_lab = np.array([[37.54,14.37,14.92],[62.73,35.83,56.5],[28.37,15.42,-49.8],
                                    [95.19,-1.03,2.93],[64.66,19.27,17.5],[39.43,10.75,-45.17],
                                    [54.38,-39.72,32.27],[81.29,-0.57,0.44],[49.32,-3.82,-22.54],
                                    [50.57,48.64,16.67],[42.43,51.05,28.62],[66.89,-0.75,-0.06],
                                    [43.46,-12.74,22.72],[30.1,22.54,-20.87],[81.8,2.67,80.41],
                                    [50.76,-0.13,0.14],[54.94,9.61,-24.79],[71.77,-24.13,58.19],
                                    [50.63,51.28,-14.12],[35.63,-0.46,-0.48],[70.48,-32.26,-0.37],
                                    [71.51,18.24,67.37],[49.57,-29.71,-28.32],[20.64,0.07,-0.46]])
    file = './Images_simp/xrite.jpg'
    XR,XG,XB, XR_hi,XG_hi,XB_hi, XR_lo,XG_lo,XB_lo = CC_RGB(xc_ll_x, xc_ll_y, wid_x,hei_x,file)
    # Reference data is in different order (from upper left to lower left, upper
    # 2nd left to lower 2nd left...). 
    for k in range(num_pics):
        # Crop the image 
        file_k = f"./Images_simp/{names[k]}"
        drop = RRC(file_k, crop_params, 'cv2')
        # Calibrate the image 
        drop_rgb_cal, drop_lab_cal,CC,sample,wa = color_calibration(drop, xc_ll_x, xc_ll_y, wid_x,hei_x, file,XR,XG,XB, XR_hi,XG_hi,XB_hi,XR_lo,XG_lo,XB_lo, reference_CC_lab)
        # Store Calibrated Images 
        drops_rgb_cal_x.append([drop_rgb_cal])
        drops_lab_cal_x.append([drop_lab_cal])
        CC_out.append([CC])
        sample_out.append([sample])
        WA_out.append([wa])
        if verbose:
            print(f'DONE {k+1}/{num_pics}')

    print('DONE Results')
    return drops_rgb_cal_x, drops_lab_cal_x, CC_out, sample_out, WA_out

# Define sample Class
class Sample:
    def __init__(self, crop_params, crop_img_pil, sample_ID,drop_IDs,drops, Number_of_drops, water, img_erode,
                 crop_image_erode, PIL_crop_image_erode, img_name, Notes, PTL):
        # drops is a 3D array with the eroded droplet image in each layer
        # SAMPLE_ID is a string with the sample ID/name assigned in the spreadsheet 
        # drops_IDs is an array with each of the droplet numbers in it 
        # drops is a 3D array where each layer is zero everywhere and except the location of a single droplet (comes from
        # cv2)
        # Water is the watershed result
        # img_erode is a 2D array with the eroded watershed result 
        # crop_image_erode and PIL_crop_image_erode are images with the watershed droplets super imposed on the cropped 
        # image 
        # crop_image_erode is estracted by CV2 format and PIL_crop_image_erode is extracted by PIL
        # img_name is the name of the image used for watershed 
        # Notes are any experimental Notes 
        # PTL is the pixel to lenght conversion factor 
        
        self.ID = sample_ID
        self.drops = drops
        self.drop_IDs = drop_IDs
        self.Number_of_drops = Number_of_drops
        self.img_erode = img_erode
        self.cv2_eroded = crop_image_erode
        self.PIL_eroded = PIL_crop_image_erode
        self.water = water 
        self.img_name = img_name
        self.Notes = Notes
        self.PTL = PTL
        self.crop_img_pil = crop_img_pil
        self.crop_params = crop_params 
    
    def save(self):
        # save all the sample information in a pickle format 
        pickle.dump(self.ID, open('./Sample_set_simple/sample_ID','wb'))
        pickle.dump(self.drops, open('./Sample_set_simple/drops','wb'))
        pickle.dump(self.drop_IDs, open('./Sample_set_simple/drop_IDs','wb'))
        pickle.dump(self.Number_of_drops, open('./Sample_set_simple/Number_of_drops','wb'))
        pickle.dump(self.img_erode, open('./Sample_set_simple/img_erode','wb'))
        pickle.dump(self.cv2_eroded, open('./Sample_set_simple/crop_image_erode','wb'))
        pickle.dump(self.PIL_eroded, open('./Sample_set_simple/PIL_crop_image_erode','wb'))
        pickle.dump(self.water, open('./Sample_set_simple/water','wb'))
        pickle.dump(self.img_name, open('./Sample_set_simple/img_name','wb'))
        pickle.dump(self.Notes, open('./Sample_set_simple/Notes','wb'))
        pickle.dump(self.PTL, open('./Sample_set_simple/PTL','wb'))
        pickle.dump(self.crop_params, open('./Sample_set_simple/crop_params','wb'))
        print('Saved sample data')
        
        
def point_ave(min_dis,pt_xy, verbose):
    ## Get rid of points that are too close to each other
    ### Find the points that are too close to each other 
    dis = np.empty(shape=(len(pt_xy),len(pt_xy)))
    I=[]
    J=[]
    pt_x = np.array(pt_xy[:,0])
    pt_y = np.array(pt_xy[:,1])
    for i in range(len(pt_x)):
        for j in range(len(pt_x)):
            if j!=i:
                dis[i,j]=np.sqrt((pt_x[i]-pt_x[j])**2 + (pt_y[i]-pt_y[j])**2)
                if dis[i,j]<min_dis:
                    I = np.append(I,i)
                    J = np.append(J,j)
            if j==i:
                dis[i,j]=0
    I = np.reshape(I,(len(I),1))
    J = np.reshape(J,(len(J),1))
    indices = np.concatenate((I,J),1)
    #print("Indices")
    #print(indices)
    #print(indices)
    ### Get rid of repeated points (ie. [1,2]=[2,1])
    #### Use the same procedure as above, turn into a string to compare than turn back into numbers
    cl_pts = []
    cl_1 = []
    cl_2 = []
    for i in range(len(I)):
        cl_pt= str(indices[i][0].astype('uint8'))+str(indices[i][1].astype('uint8'))
        cl_pt_rev = str(indices[i][1].astype('uint8'))+str(indices[i][0].astype('uint8'))
        if cl_pt not in cl_pts and cl_pt_rev not in cl_pts:
            cl_1 = np.append(cl_1, indices[i][0].astype('uint8'))
            cl_2 = np.append(cl_2, indices[i][1].astype('uint8'))
            cl_pt_arr = [cl_pt]
            cl_pts = np.concatenate((cl_pts,cl_pt_arr),0)
    #print('Cl_pts')
    #print(cl_pts)
    cl_pts_num = np.empty((len(cl_pts),2))
    for i in range(len(cl_pts)):
        for j in range(2):
            if j ==0:
                cl_pts_num[i][j]=cl_1[i]
            else:
                cl_pts_num[i][j]=cl_2[i]
    #print("cl_pts_num")
    #print(cl_pts_num)
    ### Average the two points that are too close 
    for i in range(len(cl_pts_num)):
        pt1x = pt_xy[int(cl_pts_num[i][0])][0]
        pt1y = pt_xy[int(cl_pts_num[i][0])][1]
        pt2x = pt_xy[int(cl_pts_num[i][1])][0]
        pt2y = pt_xy[int(cl_pts_num[i][1])][1]
        ptx = (pt1x+pt2x)/2
        pty = (pt1y+pt2y)/2
        if i == 0: 
            cor = [[ptx,pty]]
        else:
            cor = np.concatenate((cor, [[ptx,pty]]),0)
    ### Create the final list of points
    # Add another filter, if you have a bunch of points that are too close together after averaging then chose just one of those points 
    # print('cor')
    # print(cl_pts_num)
    # print(cor)
    dis = np.empty(shape=(len(cor),len(cor)))
    I=[]
    J=[]
    pt_x = np.array(cor[:,0])
    pt_y = np.array(cor[:,1])
    for i in range(len(pt_x)):
        for j in range(len(pt_x)):
            if j!=i:
                dis[i,j]=np.sqrt((pt_x[i]-pt_x[j])**2 + (pt_y[i]-pt_y[j])**2)
                if dis[i,j]<min_dis:
                    I = np.append(I,i)
                    J = np.append(J,j)
            if j==i:
                dis[i,j]=0
    I = np.reshape(I,(len(I),1))
    J = np.reshape(J,(len(J),1))
    indices = np.concatenate((I,J),1)
    points_TBD = np.unique(indices)
    #print("TBD")
    #print(points_TBD)
    #print("cor")
    #print(cor)
    cor = np.delete(cor,points_TBD.astype('uint8')[:-1],0)
    
    cor_rem = np.unique(cl_pts_num)
    #print('cor_rem = np.unique(cl_pts_num)')
    #print(cor_rem)
    pts_final = np.copy(pt_xy)
    #print(' pts_final = np.copy(pt_xy)')
    #print(pts_final)
    pts_final = np.delete(pts_final,cor_rem.astype('uint8'),0)
    #print('pts_final = np.delete(pts_final,cor_rem.astype(uint8),0)')
    #print(pts_final)
    
    pts_final = np.concatenate((pts_final,cor),0)
    #print('pts_final = np.concatenate((pts_final,cor),0)')
    #print(pts_final)
    return cor, cl_pts_num,pts_final

def Edge_detect(crop_image,ap_size,k,lb,ub,c,block,ts,srn,stn,verbose):
    # Image Processes 
    ## Copy the image to be warped 
    im = np.copy(crop_image)
    im2= np.copy(crop_image)
    ## Image processing RGB>GRAY>threshold>edges 
    kernel1 = np.ones((k,k), np.uint8)
    imer = cv2.dilate(im,kernel1)
    imer = cv2.erode(imer,kernel1)
    imgray = cv2.cvtColor(imer, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)
    #kernel = np.array([[0, -1, 0], [-1, 4.5, -1], [0, -1, 0]])
    # Sharpen the image
    #im_sharp = cv2.filter2D(imgray, -1, kernel)
    #im_sharp = cv2.dilate(im_sharp,kernel1)
    #ret, thresh1 = cv2.threshold(im_sharp, ta, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(thresh, lb, ub,apertureSize=ap_size)
    '''
    imer = cv2.erode(im,(5,5))
    imgray = cv2.cvtColor(imer, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(im, ta, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(imgray, a, b,apertureSize=ap_size)
    '''
    ## Use probalistic Hough Lines transfrom to find the lines in the image that shoudl correspond to the edge of the glass slide 
    linesP = cv2.HoughLinesP(edged, 1, np.pi / 180, ts, None, srn,stn)
    ## Initiate empty vectors to store the slopes and y-intercepts of all the detected lines
    M = []
    B = []
    if linesP is not None:
        # for each line find the end points and calculate to lines slope (m) and y-intercept(b)
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(im2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            m = (l[3]-l[1])/(l[2]-l[0])
            b = -m*l[0] + l[1]
            M = np.append(M,[m])
            B = np.append(B,[b]) 

    ## Define the figure for visual validation of computer vision algorythm 
    fig, ax = plt.subplots(3,figsize=(1248 / 200, 1024 / 300))
    ## Plot the lines 
    for i in range(len(M)):
        ax[0].axline((0,B[i]),slope=M[i])
    ax[0].imshow(im)
    ax[0].title.set_text('Detected Edges of the Glass Slide')
    ax[1].imshow(im2)
    ax[1].title.set_text('Results of Hough Transform')
    ax[2].imshow(edged)
    ax[2].title.set_text('Results of Canny Edge Detection')
    ax[0].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    ax[1].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    ax[2].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    fig.tight_layout()

    # Use the results of Image Processing to find the corners of the glass slide 
    max_x = np.size(crop_image,1)
    max_y = np.size(crop_image,0)
    pts_x = np.empty(shape=(len(M),len(M)))
    pts_y = np.empty(shape=(len(M),len(M)))
    pt_x=[]
    pt_y=[]
    ## find all interceps between the lines and plot the intersections
    for i in range(len(M)):
        for j in range(len(M)):
            # Between every line (ie line 1 and 3) find the x(pts_x) and y(pts_y) intercept of the lines 
            if j != i:
                pts_x[i,j] = (B[j]-B[i])/(M[i]-M[j])
                pts_y[i,j] = M[i]*pts_x[i,j] + B[i]
                # If the points are within the image, save them as potential corner points and plot them
                if pts_x[i,j]<=max_x and pts_y[i,j]<=max_y and pts_x[i,j]>0 and pts_y[i,j]>0:
                    ax[0].plot(pts_x[i,j],pts_y[i,j], marker="o", markersize=5, markeredgecolor="red",markerfacecolor="red")
                    pt_x = np.append(pt_x,pts_x[i,j])
                    pt_y = np.append(pt_y,pts_y[i,j])
            else:
                pts_x[i,j] = 0
                pts_y[i,j] = 0
    ## Make sure the detected edges and points are ploted on a figure that is the same size as the image to be warped
    ax[0].set_xlim(0,np.size(crop_image,1))
    ax[0].set_ylim(0,np.size(crop_image,0))
    ax[0].invert_yaxis()
    ## Get rid of repeated points in pt_x and pt_y (ie the intercept between line 1&3 is that same as betweene 3&1)
    pt_x = np.reshape(pt_x,(len(pt_x),1))
    pt_y = np.reshape(pt_y,(len(pt_y),1))
    ### turn the points into strings and find unique points
    pt_xys = []
    for i in range(len(pt_x)):
        pt_xs = str(pt_x[i])
        pt_ys = str(pt_y[i]) 
        pt_s = pt_xs + ',' + pt_ys
        if pt_s not in np.unique(pt_xys): 
            pt_xys.append(pt_s)
    ###To check that only unique points are detected uncomment the print statement below
    #print(pt_xys)
    # turn it back into numbers
    pt_xy=[[0,0]]
    for i in range(len(pt_xys)):
        # Split the points into x and y 
        pt_xspl,pt_yspl = np.array(pt_xys[i].split(','))
        # get rid of the brackets
        pt_xspl = pt_xspl[1:-1]
        pt_yspl = pt_yspl[1:-1]
        # Turn it into an float>array>reshape to allow for concatenation 
        pt_xspl = np.array(float(pt_xspl)).reshape((1,1))
        pt_yspl = np.array(float(pt_yspl)).reshape((1,1))
        # Concatenate and reshape the x y points in to a array like [x y]
        pt_xyspl = np.concatenate((pt_xspl,pt_yspl),0)
        pt_xyspl = np.reshape(pt_xyspl,(1,2))
        # Add them to the list of unique points now all as floats
        pt_xy = np.concatenate((pt_xy,pt_xyspl),0)
    ### Get rid of the first 0,0 point added for ease of processing 
    pt_xy=pt_xy[1:] 
    if verbose:
        plt.show()
    plt.close()
    return pt_xy

def warp_image(corners,crop_image,img_erode):
    # Find which of the points is the lower left, upper right etc..
    # Find bottom two which have the highest y values 
    sorting = np.copy(corners)
    sort = sorted(sorting,key=lambda x:x[1],reverse=True)
    sort = np.reshape(sort,(4,2)) 
    bottom = sort[:-2]
    top = sort[-2:]
    bottom_sort = sorted(bottom,key=lambda x:x[0],reverse=True)
    bottom_sort = np.reshape(bottom_sort,(2,2))
    top_sort = sorted(top,key=lambda x:x[0],reverse=True)
    top_sort = np.reshape(top_sort,(2,2))
    # within that find the one with the lowest x = ll and the higher x = lR
    ll = bottom_sort[1]
    lr = bottom_sort[0]
    ul = top_sort[1]
    ur = top_sort[0]
    # print(ll)
    # Define destination of each corner [[ll],[lr],[ul],[ur]]
    w = np.size(crop_image,1)
    h = np.size(crop_image,0)
    dest = np.float32([[0,h],[w,h],[0,0],[w,0]])
    pts1=np.concatenate((ll.reshape(1,2),lr.reshape(1,2),ul.reshape(1,2),ur.reshape(1,2)),0)
    M = cv2.getPerspectiveTransform(np.float32(pts1),dest)
    warp = cv2.warpPerspective(img_erode,M,(w,h))
    return warp

def clean_warp(warp,drop_IDs,Number_of_drops,img_erode):# Clean Up the Warped Image 
    # Step one get rid of any pixels with values outside the droplet IDs
    warp_clean1 = np.copy(warp)
    mask = np.isin(warp_clean1, drop_IDs)
    warp_clean1[np.where(mask == False)] = 0 
    # Step two Erode and dilate the droplets to get rid of edge pixels with different values 
    k = 3
    kernel = np.ones((k, k), np.uint8)
    kernel2 = np.ones((2*k, 2*k), np.uint8)
    warp_clean2 = cv2.erode(warp_clean1,kernel)
    warp_clean2 = cv2.dilate(warp_clean2,kernel2)
    # Step 3 For each droplet ID isolate the pixels with that ID then erode and dilate the image to get ride of noise
    # Creatd a 3D array where each layer is the warped image with the drop_ID pixels isolated 
    deconst_warp = np.empty((Number_of_drops,np.size(img_erode,0),np.size(img_erode,1)))
    # print(np.shape(deconst_warp))
    i = -1
    for ids in np.unique(drop_IDs):
                            i+=1
                            layer=np.copy(warp_clean2)
                            layer[np.where(layer!=ids)]=0 
                            layer = cv2.erode(layer,kernel2)
                            layer = cv2.dilate(layer,kernel2)
                            deconst_warp[i,:,:] = layer 
    # Step 4 combine all the individual images into one final warped and cleaned image.
    final_warp = np.sum(deconst_warp, axis = 0)
    return(final_warp,deconst_warp,warp_clean2)