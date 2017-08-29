'''
referred from: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
referred from: sentdex from youtube
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
import dicom
import cv2
import os

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INPUT_FOLDER = 'G:/DL/Lung-Cancer_Detection/sample_images/'
dimension = 224

patients = os.listdir(INPUT_FOLDER)
labels = pd.read_csv('G:/DL/Lung-Cancer_Detection/stage1_labels.csv/stage1_labels.csv')
print(labels.head())
patients.sort()	# sorting by names will always give same order of files
print(patients)

def print_slice(slice):
    for arr in slice:
        print(arr)

for patient in patients[:1]:
    # label = labels.get_value(patient, 'cancer')

    path = INPUT_FOLDER + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    '''
        slices contain n number of slices, where n can be around 200.
            Each slice in slices contain a number of non-homogenous attributes
                one of the attribute is pixel_array
                pixel_array is a numpy array with shape (512, 512), and it contains the actual data (the image)
    '''

    # slice_thickness is missing. therefore we infer it indirectly
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for slice in slices:
        slice.SliceThickness = slice_thickness

    print(len(slices))
    print(slices[0].pixel_array.shape)

    # resizing scan images from (512, 512) to (dimension, dimension)
    image_slices = []
    for slice in slices:
        image_slices.append(cv2.resize(np.array(slice.pixel_array), (dimension, dimension)))
    image_slices = np.array(image_slices)

    # since the scans are circles, but images are square, the black portion outside circle corresponds to -2000 (air)
    # replace that with 0 (water)
    image_slices[image_slices == -2000] = 0

    # print_slice(slices[0])
    print(image_slices[0])
    plt.imshow(image_slices[100])
    plt.show()

    '''Convert to Hounsfield units (HU)'''
    # for slice_number in range(len(slices)):
    #     intercept = slices[slice_number].RescaleIntercept
    #     slope = slices[slice_number].RescaleSlope
    #
    #     if slope != 1:
    #         print('fdslfdslf')
    #         image_slices[slice_number] = slope * image_slices[slice_number].astype(np.float64)
    #         image_slices[slice_number] = image_slices[slice_number].astype(np.int16)
    #
    #     image_slices[slice_number] += np.int16(intercept)
    #
    # print(image_slices[0])
    # plt.imshow(image_slices[100])
    # plt.show()

