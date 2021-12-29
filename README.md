# Assessing change in glaciers with machine learning

Contributors: Maxwell Bergström, Sélène Ledain, Raphaël Miazza\
Image Processing for Earth Observation ENV-540\
Fall 2021 - Final Project\

## About

Studying glaciers is essential in monitoring climate change and understanding the role that humans play in accelerating global warming. This study aims at developing semantic segmentation models, which are based on ensemble methods and namely decision trees, to map the McBride glacier in Alaska, known to have experienced an important shrinking. By outputting "glacier" or "not glacier", the change in surface could be calculated. Furthermore, the models could provide new insights on the importance of features for glacier monitoring as such information is inherent to decision trees. This could also help assess other existing methods and the features that are currently used. The two models proposed and compared are a Random Forest and a Gradient Boosting model.

## Project directory 

:file_folder: 
  |--- **GB_glacier.ipynb**: Notebook containing developed Gradient Boosting models.\
  |--- **RF_glacier.ipynb**: Notebook containing developed Random Forest models.\
  |--- **helper_functions.py**: Script imported as a module in the notebooks, containing functions used.\
  |--- **data**: 
  
 The data folder contains the initial data used in both notebooks
:file_folder: **data**

  |--- **EO_Browser_images-date**: Contains data for that date.\
  |--- **EO_Browser_images_KLI-date**: Contains data on the Klinaklini glacier for that date.\
  |--- **gt**: Contains the ground truth files named `GT_marker_date.tif` or `GT_KLI_date.tif` for the Klinaklini glacier.\
  
Each of the **EO_Browser_images-date** contains the following raw images:\
* B02.tiff
* B03.tiff
* B04.tiff
* B8A.tiff
* B11.tiff
* B12.tiff
* SWIR.tiff
* True_color.tiff


## Data

Data was downloaded from the [Sentinel Hub EO Browzer](https://apps.sentinel-hub.com/eo-browser/). The ground truth were hand labelled from images using QGIS.

The data necessary to run the project can be downloaded [here.](https://drive.google.com/drive/folders/1B_3tv_uJuDsumA87xexR0jrAJXqiS6Br?usp=sharing). Rename this folder as "data" and place it as indicated in the **Project directory** above to start running the code.


## How to run the code 

Specific models requires : 

* numpy
* skimage
* sklearn
* matplotlib
* opencv
* os
* rasterio
* warnings 
* pandas
* seaborn

If the project directory is organized as indicated, then the code can be run.

The `RF_glacier.ipynb` notebook should be run first, as images a processed and features are created and saved locally. These same feautres are then used in the `GB_glacier.ipynb` notebook.
