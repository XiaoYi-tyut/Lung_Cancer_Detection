import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import cv2
import os

from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, merge, ZeroPadding3D, AveragePooling3D, Flatten, Dropout, Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

number_of_classes = 2
dimension = 112
number_of_channels = 1
batch_size = 50
features_directory = '/input/preprocessed_images/'
labels_directory = '/input/stage1_labels.csv/'

def load_features_and_labels_dataset(features_directory, labels_directory):
    dataset_features = []
    dataset_labels = []

    features_file_names = os.listdir(features_directory)
    features_file_names.sort()
    label_file = os.listdir(labels_directory)
    label_file = pd.read_csv(labels_directory + label_file[0], index_col=0)

    files_read = 0
    files_no_labels = 0
    for patient in features_file_names:
        try:
            label_index = label_file.get_value(patient[:-4], 'cancer')  #[:-4} is to remove .npy from file name
            dataset_feature_temp = np.load(features_directory + patient).reshape(20, dimension, dimension, number_of_channels)
            dataset_features.append(dataset_feature_temp)
            label = np.zeros(2)
            label[int(label_index)] = 1
            dataset_labels.append(label)

            # for rebalancing:
            if int(label_index) == 1:
                dataset_labels.append(label)
                dataset_labels.append(label)
                dataset_labels.append(label)
                dataset_features.append(dataset_feature_temp)
                dataset_features.append(dataset_feature_temp)
                dataset_features.append(dataset_feature_temp)

            files_read += 1
            print('files_read:', files_read)
        except:
            print('cannot find label for patient or cannot reshape due to shape imbalance', patient)
            files_no_labels += 1

    print('files_no_labels:', files_no_labels)
    return np.array(dataset_features), np.array(dataset_labels)

def duplicate_d_times(dataset, duplicate_times):
    dataset_concatenated = dataset
    for d in range(1, duplicate_times):
        dataset_concatenated = np.concatenate((dataset_concatenated, dataset), axis=0)

    return dataset_concatenated

dataset_features, dataset_labels = load_features_and_labels_dataset(features_directory, labels_directory)
print('dataset_features.shape:', dataset_features.shape)
print('dataset_labels.shape:', dataset_labels.shape)

# reshaping due to issues not able to find
dataset_features = dataset_features.reshape(-1, 20, dimension, dimension, number_of_channels)

# divide
dataset_test_features = dataset_features[2200:dataset_features.shape[0]]
dataset_test_labels = dataset_labels[2200:dataset_labels.shape[0]]
dataset_train_features = dataset_features[0:2200]
dataset_train_labels = dataset_labels[0:2200]

# duplicate training dataset d times to increase its size
d = 2
dataset_train_features = duplicate_d_times(dataset_train_features, d)
dataset_train_labels = duplicate_d_times(dataset_train_labels, d)

# print shape
print('dataset_train_features.shape:', dataset_train_features.shape)
print('dataset_train_labels.shape:', dataset_train_labels.shape)
print('dataset_test_features.shape:', dataset_test_features.shape)
print('dataset_test_labels.shape:', dataset_test_labels.shape)

ones = 0
for arr in dataset_test_labels:
    if arr[1] == 1:
        ones += 1

print('ones:', ones)

# neural network start:
input = Input(shape=(20, dimension, dimension, number_of_channels))
conv1 = Convolution3D(128, 3, 3, 3, subsample=(2,2,2), border_mode='same', activation='relu', W_regularizer=l2(0.0002))(input)
conv1 = ZeroPadding3D(padding=(1,1,1))(conv1)
conv1 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(conv1)

conv1 = BatchNormalization()(conv1)
conv2 = Convolution3D(256, 3, 3, 3, subsample=(2,2,2), border_mode='same', activation='relu', W_regularizer=l2(0.0002))(conv1)
conv2 = ZeroPadding3D(padding=(1,1,1))(conv2)
conv2 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(conv2)

conv2 = BatchNormalization()(conv2)
conv3 = Convolution3D(512, 3, 3, 3, subsample=(2,2,2), border_mode='same', activation='relu', W_regularizer=l2(0.0002))(conv2)
conv3 = ZeroPadding3D(padding=(1,1,1))(conv3)
conv3 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(conv3)

conv3 = BatchNormalization()(conv3)
conv4 = Convolution3D(512+256, 3, 3, 3, subsample=(2,2,2), border_mode='same', activation='relu', W_regularizer=l2(0.0002))(conv3)
conv4 = ZeroPadding3D(padding=(1,1,1))(conv4)
conv4 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(conv4)

conv4 = BatchNormalization()(conv4)
flatten = Flatten()(conv4)
fc = Dense(1024, activation='relu', name='fc')(flatten)
fc = Dropout(0.5)(fc)

fc = BatchNormalization()(fc)
output_layer = Dense(number_of_classes, name='output_layer')(fc)
output_layer = Activation('softmax')(output_layer)

epochs = 200
lrate = 0.00001
decay = lrate/epochs
adam = Adam(decay=decay)

# checkpoint
filepath = "/output/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model = Model(inputs=input, outputs=output_layer)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(dataset_train_features, dataset_train_labels, validation_data=(dataset_test_features, dataset_test_labels), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
# predictions = model.predict(dataset_test_features)

model_json = model.to_json()
with open("/output/model_g_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/output/model_g_1.h5")
print("Saved model to disk")
