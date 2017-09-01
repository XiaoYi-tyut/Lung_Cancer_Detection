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

INPUT_FOLDER = 'G:/DL/Lung-Cancer_Detection/stage1/stage1/'
OUTPUT_FOLDER = 'G:/DL/Lung-Cancer_Detection/preprocessed_images/'
dimension = 112
minimum_bound = -1000.0
maximum_bound = 400.0
num_slices = 20

patients = os.listdir(INPUT_FOLDER)
patients.sort()	# sorting by names will always give same order of files
labels = pd.read_csv('G:/DL/Lung-Cancer_Detection/stage1_labels.csv/stage1_labels.csv')

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

def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, level=threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


for patient in patients[:1]:
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

    image_slices = []
    for slice in slices:
        slice = np.array(slice.pixel_array)
        slice[slice == -2000] = 0
        image_slices.append(slice)

    image_slices = np.array(image_slices)

    '''Convert to Hounsfield units (HU)'''
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image_slices[slice_number] = slope * image_slices[slice_number].astype(np.float64)
            image_slices[slice_number] = image_slices[slice_number].astype(np.int16)

        image_slices[slice_number] += np.int16(intercept)


    # lung segmentation mask:
    segmented_lungs_fill = segment_lung_mask(image_slices, True)

    # apply lung segmentation mask to all images in image_slices
    image_slices = np.multiply(image_slices, segmented_lungs_fill)

    image_slices = np.multiply(image_slices, [-1])

    # resizing scan images from (512, 512) to (dimension, dimension)
    new_image_slices = []
    for image_slice in image_slices:
        temp = cv2.resize(np.array(image_slice), (dimension, dimension))
        new_image_slices.append(temp)

    image_slices = np.array(new_image_slices)

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

    try:
        # label = labels.get_value(patient, 'cancer')
        np.save(OUTPUT_FOLDER + patient, image_slices)
    except:
        print('label not found for ', patient)