import tensorflow as tf
import numpy as np
import csv
import cv2
import os

from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, merge, ZeroPadding3D, AveragePooling3D, Flatten, Dropout, Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from LRN_helper import LRN2D

number_of_classes = 2
dimension = 224
number_of_channels = 1
batch_size = 50
features_directory = 'G:/DL/Lung-Cancer_Detection/preprocessed_images/'
labels_directory = 'G:/DL/Lung-Cancer_Detection/???????????????/'

def load_features_and_labels_dataset(features_directory, labels_directory):
    dataset_features = []
    dataset_labels = []

    features = os.listdir(features_directory)
    features.sort()
    labels = os.listdir(labels_directory)
    labels.sort()

    for file in labels:
        try:
            label = labels.get_value(labels_directory + file, 'cancer')
            dataset_labels.append(label)
            dataset_features.append(np.load(features_directory + file))
        except:
            print('label missing for file', file)

    return np.array(dataset_features), np.array(dataset_labels)




    return np.array(dataset_features)

def inception(input, prefix, n1x1, r3x3, n3x3, r5x5, n5x5, m1x1):
    # input = Input(shape=shape)(input)
    layer_conv_1x1_b = Convolution3D(r3x3, 1, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_b', W_regularizer=l2(0.0002))(input)
    layer_conv_1x1_b= BatchNormalization()(layer_conv_1x1_b)
    layer_conv_1x1_c = Convolution3D(r5x5, 1, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_c', W_regularizer=l2(0.0002))(input)
    layer_conv_1x1_c = BatchNormalization()(layer_conv_1x1_c)
    layer_max_3x3_d = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same', name=prefix+'layer_max_3x3_d')(input)

    layer_conv_1x1_a = Convolution3D(n1x1, 1, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_a', W_regularizer=l2(0.0002))(input)
    layer_conv_1x1_a = BatchNormalization()(layer_conv_1x1_a)
    layer_conv_3x3_b = Convolution3D(n3x3, 3, 3, 3, border_mode='same', activation='relu', name=prefix+'layer_conv_3x3_b', W_regularizer=l2(0.0002))(layer_conv_1x1_b)
    layer_conv_3x3_b = BatchNormalization()(layer_conv_3x3_b)
    layer_conv_5x5_c = Convolution3D(n5x5, 5, 5, 5, border_mode='same', activation='relu', name=prefix+'layer_conv_5x5_c', W_regularizer=l2(0.0002))(layer_conv_1x1_c)
    layer_conv_5x5_c = BatchNormalization()(layer_conv_5x5_c)
    layer_conv_1x1_d = Convolution3D(m1x1, 1, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_d', W_regularizer=l2(0.0002))(layer_max_3x3_d)
    layer_conv_1x1_d = BatchNormalization()(layer_conv_1x1_d)

    output = merge([layer_conv_1x1_a, layer_conv_3x3_b, layer_conv_5x5_c, layer_conv_1x1_d], mode='concat')
    return output

dataset_features, dataset_labels = load_features_and_labels_dataset(features_directory, labels_directory)

# divide
dataset_test_features = dataset_features[1200:dataset_features.shape[0], :]
dataset_test_labels = dataset_labels[1200:dataset_labels.shape[0], :]
dataset_train_features = dataset_features[0:1200, :]
dataset_train_labels = dataset_labels[0:1200, :]

# reshaping
dataset_train_features = dataset_train_features.reshape((-1, dimension, dimension, number_of_channels))
dataset_test_features = dataset_test_features.reshape((-1, dimension, dimension, number_of_channels))
print('dataset_test_features.shape:', dataset_test_features.shape)
print('dataset_test_labels.shape:', dataset_test_labels.shape)

# neural network start:
input = Input(shape=(dimension, dimension, number_of_channels))
conv1 = Convolution3D(64, 7, 7, 7, subsample=(2,2,2), border_mode='same', activation='relu', W_regularizer=l2(0.0002))(input)
conv1 = ZeroPadding3D(padding=(1,1,1))(conv1)
conv1 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(conv1)
# conv1 = LRN2D()(conv1)

conv1 = BatchNormalization()(conv1)
conv2 = Convolution3D(64, 1, 1, 1, border_mode='same', activation='relu', W_regularizer=l2(0.0002))(conv1)
conv2 = Convolution3D(192, 3, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(0.0002))(conv2)
# conv2 = LRN2D()(conv2)
conv2 = ZeroPadding3D(padding=(1,1,1))(conv2)
conv2 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(conv2)

conv2 = BatchNormalization()(conv2)
inception1 = inception(conv2, '3a', 64, 96, 128, 16, 32, 32)

inception1 = BatchNormalization()(inception1)
inception2 = inception(inception1,'3b', 128, 128, 192, 32, 96, 64)
inception2 = ZeroPadding3D(padding=(1,1,1))(inception2)
inception2 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(inception2)

inception2 = BatchNormalization()(inception2)
inception3 = inception(inception2, '4a', 192, 96, 208, 16, 48, 64)

inception3 = BatchNormalization()(inception3)
inception4 = inception(inception3, '4b', 160, 112, 224, 24, 64, 64)

inception4 = BatchNormalization()(inception4)
inception5 = inception(inception4, '4c', 128, 128, 256, 24, 64, 64)

inception5 = BatchNormalization()(inception5)
inception6 = inception(inception5, '4d', 112, 144, 288, 32, 64, 64)

inception6 = BatchNormalization()(inception6)
inception7 = inception(inception6, '4e', 256, 160, 320, 32, 128, 128)
inception7 = ZeroPadding3D(padding=(1,1,1))(inception7)
inception7 = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), border_mode='valid')(inception7)

inception7 = BatchNormalization()(inception7)
inception8 = inception(inception7, '5a', 256, 160, 320, 32, 128, 128)

inception8 = BatchNormalization()(inception8)
inception9 = inception(inception8, '5b', 384, 192, 384, 48, 128, 128)
# inception9 = ZeroPadding2D(padding=(1,1))(inception9)
inception9 = AveragePooling3D(pool_size=(7,7,7), strides=(1,1,1), border_mode='valid')(inception9)

inception9 = BatchNormalization()(inception9)
flatten = Flatten()(inception9)
fc = Dense(1024, activation='relu', name='fc')(flatten)
fc = Dropout(0.7)(fc)

fc = BatchNormalization()(fc)
output_layer = Dense(number_of_classes, name='output_layer')(fc)
output_layer = Activation('softmax')(output_layer)

epochs = 50
lrate = 0.0001
decay = lrate/epochs
adam = Adam(decay=decay)
# early_stopping = EarlyStopping(patience=2)

# checkpoint
filepath = "/output/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model = Model(inputs=input, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(dataset_train_features, dataset_train_labels, validation_data=(dataset_test_features, dataset_test_labels), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
predictions = model.predict(dataset_test_features)

model_json = model.to_json()
with open("/output/model_g_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/output/model_g_1.h5")
print("Saved model to disk")
