#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:29:00 2018

@author: ajoshi
"""

#AUM
#Shree Ganeshaya Namaha
#tf.test.gpu_device_name()
from scipy.ndimage.interpolation import rotate
from skimage.color import rgb2gray
from skimage.io import imread
from keras.layers import Input,Conv3D,concatenate,MaxPooling3D,Flatten,Dense,Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from keras import losses
sizeX = 128
sizeY = 128
sizeZ = 128

#taken from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:3]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size, center_square[2]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
#    M = cv2.getAffineTransform(pts1, pts2)
#    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.ndgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)



def get_rr_net():
    inputs = Input((sizeX, sizeY, sizeZ, 2))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4_1 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4_1)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    flat1 = Flatten()(conv5)
    d1= Dense(512,activation='relu')(flat1)
    d2= Dense(64,activation='relu')(d1)

    out_theta = Dense(5)(d2)
#    conv_tx = Conv2D(1, (1, 1), activation=final_activation)(conv5)
#    conv_ty = Conv2D(1, (1, 1), activation=final_activation)(conv5)
#    conv_theta = Conv2D(1, (1, 1), activation='tanh')(conv5)

#    out_img = rotate(inputs,conv_theta)

    model = Model(inputs=[inputs], outputs=out_theta)

    model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])

    return model


def gen_train_data(img, N=1024,  nodist=0):
    imgs_train = np.zeros((N, img.shape[0], img.shape[1], img.shape[2], 2))
    noise = uniform(low=-1,high=1,size=imgs_train.shape)
    out_train = np.zeros((N, 3))
    rot1 = uniform(low=-60, high=60, size=(N,1))
    rot2 = uniform(low=-60, high=60, size=(N,1))
    tx = uniform(low=-100, high=100, size=(N,1))
    ty = uniform(low=-100, high=100, size=(N,1))
    tz = uniform(low=-100, high=100, size=(N,1))
    out_train[:,0]=rot1.squeeze()
    out_train[:,1]=rot2.squeeze()
    out_train[:,2]=tx.squeeze()
    out_train[:,3]=ty.squeeze()
    out_train[:,4]=tz.squeeze()


    in_rot1 = uniform(low=-60, high=60, size=(N,1))
    in_rot2 = uniform(low=-60, high=60, size=(N,1))

    for j in range(N):
      
#        img2 = tf.warp(img,aff)
        img2 = rotate(img, angle=in_rot1[j], axes=[0,1], mode='edge',reshape=False)
        img2 -= np.mean(img2)
        img2 /= np.std(img2)
        if nodist==0:
            img2 = elastic_transform(img2, img2.shape[1] * 1.2, img2.shape[1] * 0.08,img2.shape[1] * 0.08)
        
        imgs_train[j, :, :, 0] = img2 + (1.0-nodist)*noise[j,:,:,0]
       
        aff = tf.AffineTransform(rotation = (np.pi/180.0)*np.float(rot[j]), translation=(tx[j],ty[j]))

        #img3 = 10-1*np.tanh(img2) + 0*  noise[j,:,:,1]#
        img3 = 10-1*np.tanh(tf.warp(img2, aff, mode='edge')) +  (1.0-nodist)*noise[j,:,:,1]#
        
        if nodist == 0:
            img3 = elastic_transform(img3, img3.shape[1] * .2, img3.shape[1] * 0.08,img3.shape[1] * 0.08)

        img3 -= np.mean(img3)
        img3 /= np.std(img3)
        imgs_train[j, :, :, 1] = img3
        
        if 0:
            plt.imshow(img3)
            plt.show()
            
            plt.imshow(img2)
            plt.show()
             
    return imgs_train, out_train


def train_model(img):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    img = img.astype(float)
    mean = np.mean(img)  # mean for data centering
    std = np.std(img)  # std for data normalization

    img -= mean
    img /= std
    

    print('Creating and compiling model...')
    rrmodel = get_rr_net()
#    rrmodel.load_weights('weights.h5')
    model_checkpoint = ModelCheckpoint('weights3d.h5', monitor='val_loss', save_best_only=True)

    print('Fitting Model')
    for repind in range(100):
        imgs_train, out_train = gen_train_data(img, 512)
        history = rrmodel.fit(imgs_train, out_train, batch_size=64, epochs=5, verbose=1,
                              shuffle=True, validation_split=0.2,
                              callbacks=[model_checkpoint])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model fit mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
def test_model():
    
    print('Test data...')
    img = resize(rgb2gray(imread('sample_brain.png')).astype('float32'),(img_rows,img_cols),mode='reflect')
    mean = np.mean(img)  # mean for data centering
    std = np.std(img)  # std for data normalization

    img -= mean
    img /= std

    rrmodel = get_rr_net()
    rrmodel.load_weights('weights3d.h5')

    imgs_test, out_test = gen_train_data(img, 4, nodist=1)
#    plt.imshow(np.absolute(imgs_test[0,:,:,0].squeeze()),cmap='gray')
#    plt.imshow(np.absolute(imgs_test[0,:,:,1].squeeze()),cmap='gray')

    pred_theta = rrmodel.predict(imgs_test, verbose=1)

    for jj in range(4):
        fig=plt.figure(figsize=(20,10)); 
        fig.add_subplot(1,3,1);ax = plt.gca(); ax.grid(False)
        plt.axis('off')
        plt.imshow(np.absolute(imgs_test[jj,:,:,0].squeeze()),cmap='gray')
        fig.add_subplot(1,3,2);ax = plt.gca(); ax.grid(False)
        plt.axis('off')
        plt.imshow(np.absolute(imgs_test[jj,:,:,1].squeeze()),cmap='gray')
        aff = tf.AffineTransform(rotation = (np.pi/180.0)*pred_theta[jj,0], translation=(pred_theta[jj,1],pred_theta[jj,2]))
        img2 = tf.warp(np.absolute(imgs_test[jj,:,:,0].squeeze()),aff)
        fig.add_subplot(1,3,3);ax = plt.gca(); ax.grid(False)
        plt.axis('off')
        plt.imshow(img2,cmap='gray')
        plt.show()
    
    
    print(pred_theta)
    print(out_test)



