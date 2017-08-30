'''
referred from: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
referred from: sentdex from youtube
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
import math
import dicom
import cv2
import os
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INPUT_FOLDER = 'G:/DL/Lung-Cancer_Detection/sample_images/'
dimension = 224
minimum_bound = -1000.0
maximum_bound = 400.0
num_slices = 20

patients = os.listdir(INPUT_FOLDER)
labels = pd.read_csv('G:/DL/Lung-Cancer_Detection/stage1_labels.csv/stage1_labels.csv')
patients.sort()	# sorting by names will always give same order of files

def print_slice(slice):
    for arr in slice:
        print(arr)

def convert_to_gray(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def mean(x):
    return sum(x)/len(x)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def normalize(image):
    # values below -1000 (minumum_bound) corresponds to air
    # values above 400 (maximum_bound) corresponds to bones
    # therefore we normalize between -1000 and 400, and convert them to range between 0 and 1
    image = (image - minimum_bound) / (maximum_bound - minimum_bound)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

for patient in patients[:5]:
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

    # resizing scan images from (512, 512) to (dimension, dimension)
    image_slices = []
    for slice in slices:
        temp = cv2.resize(np.array(slice.pixel_array), (dimension, dimension))
        temp[temp == -2000] = 0 # since the scans are circles, but images are square, the black portion outside circle corresponds to -2000 (air). replace that with 0 (water)
        temp = normalize(temp)
        image_slices.append(temp)
    image_slices = np.array(image_slices)

    temp = []
    chunk_sizes = math.ceil(len(image_slices) / num_slices)
    for image_slice_chunk in chunks(image_slices, chunk_sizes):
        image_slice_chunk = list(map(mean, zip(*image_slice_chunk)))
        temp.append(image_slice_chunk)

    image_slices = temp

    if len(image_slices) == num_slices - 1:
        image_slices.append(image_slices[-1])

    if len(image_slices) == num_slices - 2:
        image_slices.append(image_slices[-1])
        image_slices.append(image_slices[-1])

    if len(image_slices) == num_slices + 2:
        new_val = list(map(mean, zip(*[image_slices[num_slices - 1], image_slices[num_slices], ])))
        del image_slices[num_slices]
        image_slices[num_slices - 1] = new_val

    if len(image_slices) == num_slices + 1:
        new_val = list(map(mean, zip(*[image_slices[num_slices - 1], image_slices[num_slices], ])))
        del image_slices[num_slices]
        image_slices[num_slices - 1] = new_val

    print(len(image_slices))


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

    # print_slice(slices[0])
    # print(image_slices[0])
    # plt.imshow(image_slices[100])
    # plt.show()