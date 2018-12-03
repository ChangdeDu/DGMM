#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:50:17 2017

@author: duchangde 
"""

import os    
os.environ['THEANO_FLAGS'] = "device=gpu"  
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from sklearn import preprocessing
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend
from numpy import random
from keras import optimizers
import matlab.engine
eng=matlab.engine.start_matlab()
from keras import metrics

# Load dataset
handwriten_69=loadmat('digit69_28x28.mat')
Y_train = handwriten_69['fmriTrn']
Y_test = handwriten_69['fmriTest']
X_train = handwriten_69['stimTrn']
X_test = handwriten_69['stimTest']
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

resolution = 28
X_train = X_train.reshape([X_train.shape[0], 1, resolution, resolution])
X_test = X_test.reshape([X_test.shape[0], 1, resolution, resolution])

## Normlization
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
Y_train = min_max_scaler.fit_transform(Y_train)     
Y_test = min_max_scaler.transform(Y_test)

print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)
numTrn=X_train.shape[0]
numTest=X_test.shape[0]

# Set the model parameters and hyper-parameters
maxiter = 200
nb_epoch = 1
batch_size = 10
resolution = 28
D1 = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
D2 = Y_train.shape[1]
K = 6
C = 5
intermediate_dim = 128

#hyper-parameters
tau_alpha = 1
tau_beta = 1
eta_alpha = 1
eta_beta = 1
gamma_alpha = 1
gamma_beta = 1

Beta = 1 # Beta-VAE for Learning Disentangled Representations
rho=0.1  # posterior regularization parameter
k=10     # k-nearest neighbors
t = 10.0 # kernel parameter in similarity measure
L = 100   # Monte-Carlo sampling

np.random.seed(1000)
numTrn=X_train.shape[0]
numTest=X_test.shape[0]

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

if backend.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)


# Building the architechture
X = Input(shape=original_img_size)
Y = Input(shape=(D2,))
Y_mu = Input(shape=(D2,))
Y_lsgms = Input(shape=(D2,))
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu', name='en_conv_1')(X)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2), name='en_conv_2')(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1, name='en_conv_3')(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1, name='en_conv_4')(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu', name='en_dense_5')(flat)

Z_mu = Dense(K, name='en_mu')(hidden)
Z_lsgms = Dense(K, name='en_var')(hidden)


def sampling(args):
    
    Z_mu, Z_lsgms = args
    epsilon = backend.random_normal(shape=(backend.shape(Z_mu)[0], K), mean=0., stddev=1.0)
    
    return Z_mu + backend.exp(Z_lsgms) * epsilon

Z = Lambda(sampling, output_shape=(K,))([Z_mu, Z_lsgms])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 14 * 14, activation='relu')

if backend.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 14, 14)
else:
    output_shape = (batch_size, 14, 14, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if backend.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash_mu = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

decoder_mean_squash_lsgms= Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='tanh')

hid_decoded = decoder_hid(Z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
X_mu = decoder_mean_squash_mu (x_decoded_relu)
X_lsgms = decoder_mean_squash_lsgms (x_decoded_relu)

#define objective function
logc = np.log(2 * np.pi)
def X_normal_logpdf(x, mu, lsgms):
    lsgms = backend.flatten(lsgms)   
    return backend.mean(-(0.5 * logc + 0.5 * lsgms) - 0.5 * ((x - mu)**2 / backend.exp(lsgms)), axis=-1)

def Y_normal_logpdf(y, mu, lsgms):  
    return backend.mean(-(0.5 * logc + 0.5 * lsgms) - 0.5 * ((y - mu)**2 / backend.exp(lsgms)), axis=-1)
   
def obj(X, X_mu):
    X = backend.flatten(X)
    X_mu = backend.flatten(X_mu)
    
    Lp = 0.5 * backend.mean( 1 + Z_lsgms - backend.square(Z_mu) - backend.exp(Z_lsgms), axis=-1)     
    
    Lx =  - metrics.binary_crossentropy(X, X_mu) # Pixels have a Bernoulli distribution  
               
    Ly =  Y_normal_logpdf(Y, Y_mu, Y_lsgms) # Voxels have a Gaussian distribution
        
    lower_bound = backend.mean(Lp + 10000 * Lx + Ly)
    
    cost = - lower_bound
              
    return  cost 

DGMM = Model(inputs=[X, Y, Y_mu, Y_lsgms], outputs=X_mu)
opt_method = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
DGMM.compile(optimizer = opt_method, loss = obj)
DGMM.summary()
# build a model to project inputs on the latent space
encoder = Model(inputs=X, outputs=[Z_mu,Z_lsgms])
# build a model to project inputs on the output space
imagepredict = Model(inputs=X, outputs=[X_mu,X_lsgms])

# build a digit generator that can sample from the learned distribution
Z_predict = Input(shape=(K,))
_hid_decoded = decoder_hid(Z_predict)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
X_mu_predict = decoder_mean_squash_mu(_x_decoded_relu)
X_lsgms_predict = decoder_mean_squash_mu(_x_decoded_relu)
imagereconstruct = Model(inputs=Z_predict, outputs=X_mu_predict)

# Initialization
Z_mu = np.mat(random.random(size=(numTrn,K)))
B_mu = np.mat(random.random(size=(K,D2)))
R_mu = np.mat(random.random(size=(numTrn,C)))
sigma_r = np.mat(np.eye((C)))
H_mu = np.mat(random.random(size=(C,D2)))
sigma_h = np.mat(np.eye((C)))

tau_mu = tau_alpha / tau_beta
eta_mu = eta_alpha / eta_beta
gamma_mu = gamma_alpha / gamma_beta

Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu)
Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))

savemat('data.mat', {'Y_train':Y_train,'Y_test':Y_test})
S=np.mat(eng.calculateS(k, t))

# Loop training
for l in range(maxiter):
    print ('**************************************************iter=', l)
    # update Z
    DGMM.fit([X_train, Y_train, Y_mu, Y_lsgms], X_train,
            shuffle=True,
            verbose=2,
            epochs=nb_epoch,
            batch_size=batch_size)           
       
    [Z_mu,Z_lsgms] = encoder.predict(X_train) 
    Z_mu = np.mat(Z_mu) 
    # update B
    temp1 = np.exp(Z_lsgms)
    temp2 = Z_mu.T * Z_mu + np.mat(np.diag(temp1.sum(axis=0)))
    temp3 = tau_mu * np.mat(np.eye(K))
    sigma_b = (gamma_mu * temp2 + temp3).I
    B_mu = sigma_b * gamma_mu * Z_mu.T * (np.mat(Y_train) - R_mu * H_mu)
    # update H
    RTR_mu = R_mu.T * R_mu + numTrn * sigma_r
    sigma_h = (eta_mu * np.mat(np.eye(C)) + gamma_mu * RTR_mu).I
    H_mu = sigma_h * gamma_mu * R_mu.T * (np.mat(Y_train) - Z_mu * B_mu)
    # update R
    HHT_mu = H_mu * H_mu.T + D2 * sigma_h
    sigma_r = (np.mat(np.eye(C)) + gamma_mu * HHT_mu).I
    R_mu = (sigma_r * gamma_mu * H_mu * (np.mat(Y_train) - Z_mu * B_mu).T).T  
    # update tau
    tau_alpha_new = tau_alpha + 0.5 * K * D2
    tau_beta_new = tau_beta + 0.5 * ((np.diag(B_mu.T * B_mu)).sum() + D2 * sigma_b.trace())
    tau_mu = tau_alpha_new / tau_beta_new
    tau_mu = tau_mu[0,0] 
    # update eta
    eta_alpha_new = eta_alpha + 0.5 * C * D2
    eta_beta_new = eta_beta + 0.5 * ((np.diag(H_mu.T * H_mu)).sum() + D2 * sigma_h.trace())
    eta_mu = eta_alpha_new / eta_beta_new
    eta_mu = eta_mu[0,0] 
    # update gamma
    gamma_alpha_new = gamma_alpha + 0.5 * numTrn * D2
    gamma_temp = np.mat(Y_train) - Z_mu * B_mu - R_mu * H_mu
    gamma_temp = np.multiply(gamma_temp, gamma_temp)
    gamma_temp = gamma_temp.sum(axis=0)
    gamma_temp = gamma_temp.sum(axis=1)
    gamma_beta_new = gamma_beta + 0.5 * gamma_temp
    gamma_mu = gamma_alpha_new / gamma_beta_new
    gamma_mu = gamma_mu[0,0] 
    # calculate Y_mu   
    Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu) 
    Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))   

# reconstruct X (image) from Y (fmri)
X_reconstructed_mu = np.zeros((numTest, img_chns, img_rows, img_cols))
HHT = H_mu * H_mu.T + D2 * sigma_h
Temp = gamma_mu * np.mat(np.eye(D2)) - (gamma_mu**2) * (H_mu.T * (np.mat(np.eye(C)) + gamma_mu * HHT).I * H_mu)
for i in range(numTest):
    s=S[:,i]
    z_sigma_test = (B_mu * Temp * B_mu.T + (1 + rho * s.sum(axis=0)[0,0]) * np.mat(np.eye(K)) ).I
    z_mu_test = (z_sigma_test * (B_mu * Temp * (np.mat(Y_test)[i,:]).T + rho * np.mat(Z_mu).T * s )).T
    temp_mu = np.zeros((1,img_chns, img_rows, img_cols))
    epsilon_std = 1
    for l in range(L):
        epsilon=np.random.normal(0,epsilon_std,1)
        z_test = z_mu_test + np.sqrt(np.diag(z_sigma_test))*epsilon
        x_reconstructed_mu = imagereconstruct.predict(z_test, batch_size=1)
        temp_mu = temp_mu + x_reconstructed_mu
    x_reconstructed_mu = temp_mu / L
    X_reconstructed_mu[i,:,:,:] = x_reconstructed_mu

# visualization the reconstructed images
n = 10
for j in range(1):
    plt.figure(figsize=(12, 2))    
    for i in range(n):
        # display original images
        ax = plt.subplot(2, n, i +j*n*2 + 1)
        plt.imshow(np.rot90(np.fliplr(X_test[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstructed images
        ax = plt.subplot(2, n, i + n + j*n*2 + 1)
        plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
