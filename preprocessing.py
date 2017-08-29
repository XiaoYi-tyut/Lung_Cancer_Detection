'''
referred from: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
import dicom
import os

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INPUT_FOLDER = 'G:/DL/Lung-Cancer_Detection/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()	# sorting by names will always give same order of files
print(patients)

def load_scan_of_user(user_path):
    slices = []
    for scan in os.listdir(user_path):
        slices.append(dicom.read_file(user_path + '/' + scan))

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for slice in slices:
        slice.SliceThickness = slice_thickness

    return slices

for user_folder in patients:
    slices = load_scan_of_user(INPUT_FOLDER + user_folder)  # INPUT_FOLDER already has / in its end
    break