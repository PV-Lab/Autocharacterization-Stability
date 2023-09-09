# Automatic-Stability-Measurement
______________________________________________
<img src="./data/automated_degradation_R3.png" align = "center" width="500" />

# Table of Contents
- [Package Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Trouble_Shooting](#trouble_shooting)


# Description

This code utilizes several computer vision techniques to quickly extract and analyse accurate color vs time data of multiple chemical compounds printed using a high-throughput Multi-Material Deposition (MMD) system called Archerfish (AF) built at MIT in the Accelerated Materials Development for Sustainability under PI Tonio Buonassisi. However, due to the use of computer vision, this package can be adaptible to other materials systems of different form factors. The only component of the code specific to the MMD platform is the composition extraction, however, this is a non-critical part of the code that will not affect the band gap results. 

This code was developed by Eunice Aissi (Course 2 MIT Ph.D Entry class of 2022) in 2023 for the Accelerated Materials Development Lab for Sustainability under PI Tonio Buonassisi. This code is based on the work of Armi Tiihomen and Aleks Siemenn (Course 2 MIT PhD Entry class of 2019) 

| Folders | Description |
| ------------- | ------------------------------ |
|[Results](./Results) | An empty folder to store results in.|
|[Images_Tiff](./Images_Tiff) |A folder with tiff files of the stability images in chronological order|
|[Images_simp](./Results) |A folder with the *JPG* image of the xrite color passport called "xrite.jgp". This folder will later be populated with the individual images extracted from the tiff file|
|[Sample_set_simple](./Sample_set_simple) | A folder with an excel sheet named sample_names.xlsx with information on the Experiment Name, Start_Image_Index (the image to start the processing from, the value should reflect its position in the first tiff file of the series. ie. Start_image_idex = 30 means that the 30th image in the first tiff file is the first image to start the image analyses from) ,Pixel_To_Length(cm) Conversion, Experimental Notes, Time_Step between images, Cut is the step size of images to analyze ( ie. cut = 30 means every 39th image is color calibrated and used in the analysis), Start Composition is the Chemical Composition of the starting compound in the gradient, and End COmposition is the Chemical composition of the last copound in the gradient ( both start and end compositions are used to generate the final figures) |
|[data](./data) | a folder with images with the gcode_XY.csv and motor_speeds.txt for composition extraction|
|[figs](./figs) | an empty folder to save all the final figures|



 Files | Description |
| ------------- | ------------------------------ |
|[main.py](./main.py)| A python file to perform automatic stability measurement on the example data|
|[examples.ipynb](./examples.ipynb)| Jupyter notebook with example data to explain how to use the package. This file is interactive with lots of output figures that are used to validate each step of the process.| 
|[mainfunctions.py](./mainfunctions.py)| A python file with the high level functions for the package|
|[colorfunctions.py](./colorfunctions.py)| A python file with all the low level functions used in the [mainfunctions.py](./mainfunctions.py) file|
|[compextractorb.py](./compextractorb.py)|  A file with function used to extract AF material compositions|
|[requirements.txt](./compextractorb.py)|  a text file with all the necessary libraries to use this package|


## Algorithm Details

This algorithm cosnsists of two parts:
- **Color Calibration**: color calibrating the input images with a reference set of colors to improve the fidelity of the coloremetric data.
- **Instability Measurement Calculation**: calculating an instability index to quantify the change in color of each material over time. 

### Color Calibration

| Before Color Calibration | Post Color Calibration|
|---------------------------|-----------------------|
|<img src="./data/Img_1020.jpg" width="400" />| <img src="./data/degradation.gif" width="400" />|


At the beginning of the degradation study, an image of a reference color chart (X-Rite Colour Checker Passport; 28 reference color patches), $I_R$, is taken under the same illumination conditions as the perovskite semiconductor samples. Images at each time step, $\Omega (\Delta t)$, are transformed into L*a*b and subsequently to a stable reference color space (standard observer CIE 1931 2 degrees, standard illuminant D50) by applying a 3D-thin plate spline distortion matrix $D$ \cite{Sun2021,s120607063} defined by $I_R$ and known colors of the reference color chart:

\begin{equation}
    \label{eq:4}
    D= \begin{bmatrix}
    V\\
    O(4,3)
    \end{bmatrix}{\begin{bmatrix}
    K & P \\
    P^T & O(4,4)
    \end{bmatrix}}^{-1}
\end{equation}

 Here, $O(n,m)$ is an $n$x$m$ zero matrix, $V$ is a matrix of the color checker reference colors in the stable reference color space, $P$ is a matrix of the color checker RGB colors obtained from $I_R$, and $K$ is a distortion matrix between the color checker colors in the reference space and in $I_R$. Using the color-calibrated images and droplet pixel locations given by $\Phi$, a final array, $R(t; \widehat{X}, \widehat{Y})$ of the average color at time $t$ for perovskite semiconductor of composition FA$_{1-x}$MA$_x$PbI$_3$ is created. The color of each droplet is measured to determine a stability metric $I_c$  

**Example of an xrite image**

<img src="./data/xrite.jpg" width="400" />

### Instability Measurement Calculation 

At the beginning of the degradation study, an image of a reference color chart (X-Rite Colour Checker Passport; 28 reference color patches), $I_R$, is taken under the same illumination conditions as the perovskite semiconductor samples. Images at each time step, $\Omega (\Delta t)$, are transformed into L*a*b and subsequently to a stable reference color space (standard observer CIE 1931 2 degrees, standard illuminant D50) by applying a 3D-thin plate spline distortion matrix $D$ \cite{Sun2021,s120607063} defined by $I_R$ and known colors of the reference color chart:

\begin{equation}
    \label{eq:5}
    I_c(\widehat{X},\widehat{Y})= \sum_{R = \{r,g,b\}} \int_{0}^{T} |R(t; \widehat{X}, \widehat{Y}) - R(0;\widehat{X},\widehat{Y})| dt,
\end{equation}


 Here, $O(n,m)$ is an $n$x$m$ zero matrix, $V$ is a matrix of the color checker reference colors in the stable reference color space, $P$ is a matrix of the color checker RGB colors obtained from $I_R$, and $K$ is a distortion matrix between the color checker colors in the reference space and in $I_R$. Using the color-calibrated images and droplet pixel locations given by $\Phi$, a final array, $R(t; \widehat{X}, \widehat{Y})$ of the average color at time $t$ for perovskite semiconductor of composition FA$_{1-x}$MA$_x$PbI$_3$ is created. The color of each droplet is measured to determine a stability metric $I_c$  
 
<img src="./data/extracted_stability_trend_and_samples_R1.png" align = "center" width="500" />

# Installation 

Package installation requirements can be found in the [requirements.txt](./requirements.txt) file.

# Usage 

## Quick Start on Example Data 

A demonstration of using the automatic stability measurement package can be found in the [example.ipynb](./example.ipynb) file. The automatic stability measurement code itself can be found in the [colorfunctions.py](./colorfunctions.py) file under the `color_calibration()` definitionfor color calibration and in the [mainfunctions.py](./mainfunctions.py) file under the `extract_ic()` definition for stability index calculations.

A quicker demonstration can be obtained by using the main.py file. The user only needs to run the file and the example data will be automatically analysed. 

## For Other Applications 

Before starting, please carefully read the folders description table as it details what files should be placed in each folder in order for the code to work. Input data should take the form of a tiff file with chronological images of the material samples. We provide a test dataset in the [Images_Tiff](./Images_Tiff) folder. Our images were obtained using a Thorlabs DCC1645C camera with the infrared filter removed to increase sensitivity towards dark samples. Once the data files are input and the sample_names.xlsx file is made according to the instructions in the list of folders, the user must define a set of parameters that are explained in detail in the comments.  

For composition extraction, if the user is analyzing materials created using a HT synthesis system such as AF, they will be required to define a set of parameters to align the print pattern to the location of the droplets in order to obtain a good composition extraction. If the material system takes another form however, the color calibration and instability measurement functions can be adapted to fit the new application. 


# Trouble_Shooting 
- Make sure the Images_tiff folder is completely empty before you run the code. Sometimes the computer file management system will add an administrative file in the folder after it is created. Before running the code please open the file through the file manager and make sure it is completely empty. 
- To contact the file creator Eunice Aissi, email eunicea@mit.edu 
