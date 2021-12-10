#!/usr/bin/env python
# coding: utf-8

# In[81]:


import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras import layers

import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (Dense, 
                                     #BatchNormalization,
                                     ReLU,
                                     LeakyReLU, 
                                     Reshape, 
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)
from tensorflow.compat.v1.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from sklearn.cluster import MiniBatchKMeans
from tensorflow.python.ops.gen_batch_ops import Batch


# In[42]:


#Read in the data
TrainingXTF= tf.keras.utils.image_dataset_from_directory(
    'fashion-product-images-small/myntradataset', labels=None, label_mode=None,
    class_names=None, color_mode='rgb', batch_size=32, image_size=(60,
    80), shuffle=True, seed=123, validation_split=0.95, subset="training")


# In[5]:


#Normalize data and set batch size to 32
batch_size = 32
normalized_data = np.vstack(tfds.as_numpy(TrainingXTF.map(lambda x: (x - 127.5) / 127.5)))


# In[ ]:


#Changes batch_size to 128
batch_size = 128


# In[122]:


#Create the dicriminator neural network
#Version 2.3 adds an additional Conv2D layer 
def discriminator_model(in_shape=(60,80,3)):
    model = tf.keras.Sequential(
        [
            Conv2D(64, (3,3), padding='same', input_shape=in_shape),
            LeakyReLU(alpha=0.2),
            Conv2D(64, (3,3), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2D(64, (3,3), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3,3), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2D(256, (3,3), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation='sigmoid'),
        ]
    )
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model


# # Version 2.2 changes the parameters of one layer
# def discriminator_model(in_shape=(60,80,3)):
#     model = tf.keras.Sequential(
#         [
#             Conv2D(64, (3,3), padding='same', input_shape=in_shape),
#             LeakyReLU(alpha=0.2),
#             Conv2D(64, (3,3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2D(128, (3,3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2D(256, (3,3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Flatten(),
#             Dropout(0.4),
#             Dense(1, activation='sigmoid'),
#         ]
#     )
#     model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
#     return model
# 

# # Version 2.1 (Same as version 1)
# 
# def discriminator_model(in_shape=(60,80,3)):
#     model = tf.keras.Sequential(
#         [
#             Conv2D(64, (3,3), padding='same', input_shape=in_shape),
#             LeakyReLU(alpha=0.2),
#             Conv2D(128, (3,3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2D(128, (3,3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2D(256, (3,3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Flatten(),
#             Dropout(0.4),
#             Dense(1, activation='sigmoid'),
#         ]
#     )
#     model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
#     return model

# In[120]:


#Create the generator neural network
#Version 2.3 adds 2 Conv2DTranspose layers to the network
def generator_model(latent_dim):
    n_nodes = 256 * 15 * 20
    model = tf.keras.Sequential(
        [
            Dense(n_nodes, input_dim=latent_dim),
            LeakyReLU(alpha=0.2),
            Reshape((15, 20, 256)),
            Conv2DTranspose(64, (3, 3), strides=(1,1), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, (3, 3), strides=(1,1), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, (3, 3), strides=(1,1), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2D(3, (3, 3), padding='same', activation='tanh')
        ]
    )

    return model


# # Version 2.2 adds a Conv2DTranspose layer
# 
# def generator_model(latent_dim):
#     n_nodes = 256 * 15 * 20
#     model = tf.keras.Sequential(
#         [
#             Dense(n_nodes, input_dim=latent_dim),
#             LeakyReLU(alpha=0.2),
#             Reshape((15, 20, 256)),
#             Conv2DTranspose(64, (3, 3), strides=(1,1), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2DTranspose(128, (3, 3), strides=(1,1), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same'),
#             LeakyReLU(alpha=0.2),
#             Conv2D(3, (3, 3), padding='same', activation='tanh')
#         ]
#     )
# 
#     return model

# # Version 2.1 changes leakyReLu layers into ReLU layers 
# 
# def generator_model(latent_dim):
#     n_nodes = 256 * 15 * 20
#     model = tf.keras.Sequential(
#         [
#             Dense(n_nodes, input_dim=latent_dim),
#             ReLU(),
#             Reshape((15, 20, 256)),
#             Conv2DTranspose(128, (3, 3), strides=(1,1), padding='same'),
#             ReLU(),
#             Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same'),
#             ReLU(),
#             Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same'),
#             ReLU(),
#             Conv2D(3, (3, 3), padding='same', activation='tanh')
#         ]
#     )
# 
#     return model

# In[67]:


#Creates the adversarial network by connecting the generator and discriminator
def define_gan(g_model, d_model):
    d_model.trainable = False
    model = tf.keras.Sequential()
    model.add(g_model)
    model.add(d_model)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=.0002, beta_1=0.5))
    return model


# In[68]:


#Picks a random sample of the real images
def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


# In[69]:


# Creates latent points from the standard distribution
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# In[70]:


#Uses the latentpoints and generator model to create fake images
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


# In[71]:


#Prints the discriminators accuracy for real and fake images
def summary(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    x_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print(f'Epoch {epoch+1} summary:')
    print('Accuracy Real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    #g_model.save(f'version3_latent_models/epoch_{epoch+1}')


# In[72]:


#Trains the model and prints the discriminator losses and GAN loss
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=5, n_batch=128):
 
    batch_per_epoch = dataset.shape[0] // n_batch
    half_batch = n_batch // 2

    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            # generate real samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f"Epoch {i+1}, Batch {j+1}: " + "d1=%.3f, d2=%.3f, g=%.3f" % (d_loss1, d_loss2, g_loss))

        if (i + 1) % 5 == 0:
            summary(i, g_model, d_model, dataset, latent_dim)



# In[123]:


#Executes the existing functions
latent_dim = 250
d_model = discriminator_model()
g_model = generator_model(latent_dim)
gan_model = define_gan(g_model, d_model)
train(g_model, d_model, gan_model, normalized_data, latent_dim)


# In[134]:


#Creates and outputs generated images
x_fake, y_fake = generate_fake_samples(g_model, latent_dim, 16)
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    #im_s = generated_samples[i].detach().numpy()
    #print(im_s) 
    plt.imshow((x_fake[i]))
    #plt.imshow((x_fake[i]*255).astype(np.uint8))
    plt.xticks([])
    plt.yticks([])


# In[127]:


#Set parameters for K-means classifier

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
np.random.seed(1)
n = 10


# In[128]:


#Flattens the data
X_test_1 = normalized_data
X_test_Flat = X_test_1.reshape(len(X_test_1),-2)
print(X_test_1.shape)


# In[129]:


#Trains the K-Means classifier
kmeans = KMeans(n_clusters=n,init='random')
kmeans.fit(X_test_Flat)
Z1 = kmeans.predict(X_test_Flat)


# In[131]:


#Predicts category
prediction = kmeans.predict(X_test_Flat)
y_test = prediction


# In[132]:


#Dictionary of categories and K-means predicition
y_dict = {0:"Accessories", 1:"Shoes and Accessories", 2:"Pants", 3:"Accessories and Miscellaneous", 4:"Accessories", 5:"Dresses and Shirts", 6: "Bags", 7: "Shoes and Accessories", 8: "Dresses and Shirts", 9: "Dresses and Shirts"}


# In[135]:


#Plots fake images with their classifier label
for i in range(16):
    plt.imshow((x_fake[i]))
    plt.show()
    print(y_dict[y_test[i]])
    print(y_test[i])


# In[ ]:




