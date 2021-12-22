import numpy as np
from skimage.io import imsave, imread
from skimage.segmentation import slic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, auc, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import cv2 as cv # need to execute "pip install opencv-python" in the terminal to install module
import numpy as np
import os
import rasterio
from skimage import exposure, filters
from skimage.morphology import square
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pickle
import seaborn as sns




def get_normalized_image(image):
    """
    Rescale image to values between 0 to 255 (capping outlier values) 
    
    Parameters
    ==================
    image: Numpy array
        Image numpy array with shape (height, width, num_bands)
    
    Returns
    ==================
    output: Numpy array
        Min-max normalized image numpy array
        2D if single band
        3D if multiband
    
    """
    image = image.astype(np.float32)
    
    # if multiband
    if image.shape[2] == 3:
        output = np.zeros_like(image).astype(np.float32)
        for k in range(image.shape[2]): # for each band
            max_val = np.max(image[:, :, k])
            min_val = np.min(image[:, :, k])
            output[:, :, k] = (image[:, :, k] - min_val)/(max_val - min_val)
    
    # if single band
    else:
        image = image.reshape((image.shape[0], image.shape[1]))
        output = np.zeros((image.shape[0], image.shape[1])).astype(np.float32)
        max_val = np.max(image[:, :])
        min_val = np.min(image[:, :])
        output = (image - min_val)/(max_val - min_val)
    return output.astype(np.float32)



def NDSI(input_B3, input_B11, output_path):
    """
    Input: path to B3 band .tif and B11 .tif
    ===========
    Output: NDSI 2D numpy array
    """
    
    B3 = imread(input_B3, plugin='pil').astype(np.float32)
    B11 = imread(input_B11, plugin='pil').astype(np.float32)
    ndsi = np.zeros_like(B3)
    ndsi = (B3 - B11)/(B3 + B11)
    np.save(output_path, ndsi)
    
    
    
    
def NDWI(input_B3, input_B8, output_path):
    """
    Input: path to B3 band .tif and B8 .tif
    ===========
    Output: NDWI 2D numpy array 
    """
    
    B3 = imread(input_B3, plugin='pil').astype(np.float32)
    B8 = imread(input_B8, plugin='pil').astype(np.float32)
    ndsi = np.zeros_like(B3)
    ndwi = (B3 - B8)/(B3 + B8)
    np.save(output_path, ndwi)
    
    
    
    
def apply_filters(img, list_filters, path):
    """
    Input: list of filters to apply (use skimage names)
    ===========
    Output: Create new .npy arrays
    """
    
    for i in range(len(list_filters)):
        new_path = path
        curr_filter = list_filters[i]
        name = curr_filter.__name__
        print(name)
        new_path = new_path + '/' + name + '.npy'
        print(new_path)
        
        # Depending if function needs extra parameters
        if name == 'entropy':
            footprint = square(5)
            filtered_img = curr_filter(img, selem=footprint)
        elif name == 'autolevel':
            footprint = square(5)
            filtered_img = curr_filter(img, selem=footprint)            
        elif name == 'gabor': 
            freq = 0.6
            filtered_img = curr_filter(img, frequency = freq)
            filtered_img = filtered_img[1]
        else:
            filtered_img = curr_filter(img)
            
            
        np.save(new_path, filtered_img)

        
        
        
def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='x-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Ground Truth', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    
    
def evaluate_conf(conf_matrix):
    """
    Input: confusion matrix
    =============
    Output: metrics
    """
    
    TP = conf_matrix[1,1]
    FP = conf_matrix[0,1]
    TN = conf_matrix[0,0]
    FN = conf_matrix[1,0]
    
    acc = TP/(FP+TP)
    rec = TP/(FN+TP)
    prec = (TP + TN)/ (TP + FN + TN + FP)
    f1 = 2*prec*rec/ (prec+rec)
    
    print("Accuracy: ", acc, 
     "\nPrecision: ", prec, 
     "\nRecall: ", rec,
     "\nF1: ", f1)    

    
def compute_surface_loss(tot_area, n_pixels, diff_pixels):
    """
    Input:
    Total covered area in km^2
    Total number of pixels in the images
    The difference in piwels classified as glaciers between two dates
    ============
    Output:
    The equivalent surface lost
    """
    area_per_pix = tot_area/n_pixels
    return area_per_pix*diff_pixels 