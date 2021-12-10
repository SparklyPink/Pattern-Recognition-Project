#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install tensorflow_datasets


# In[73]:



import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras import layers

import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (Dense, 
                                     BatchNormalization, 
                                     LeakyReLU, 
                                     Reshape, 
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from sklearn.cluster import MiniBatchKMeans
from tensorflow.python.ops.gen_batch_ops import Batch


# In[74]:


#Read in the data
TrainingXTF= tf.keras.utils.image_dataset_from_directory(
    'fashion-product-images-small/myntradataset', labels=None, label_mode=None,
    class_names=None, color_mode='rgb', batch_size=32, image_size=(60,
    80), shuffle=True, seed=123, validation_split=0.95, subset="training")


# In[75]:


#Normalize data and set batch size to 32
batch_size = 32
normalized_data = np.vstack(tfds.as_numpy(TrainingXTF.map(lambda x: (x - 127.5) / 127.5)))


# In[7]:


#Create the dicriminator neural network
def discriminator_model(in_shape=(60,80,3)):
    model = tf.keras.Sequential(
        [
            Conv2D(64, (3,3), padding='same', input_shape=in_shape),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3,3), strides=(2,2), padding='same'),
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


# In[8]:


#Create the generator neural network
def generator_model(latent_dim):
    n_nodes = 256 * 15 * 20
    model = tf.keras.Sequential(
        [
            Dense(n_nodes, input_dim=latent_dim),
            LeakyReLU(alpha=0.2),
            Reshape((15, 20, 256)),
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


# In[9]:


#Creates the adversarial network by connecting the generator and discriminator
def define_gan(g_model, d_model):
    d_model.trainable = False
    model = tf.keras.Sequential()
    model.add(g_model)
    model.add(d_model)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    return model


# In[10]:


#Picks a random sample of the real images
def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


# In[11]:


# Creates latent points from the standard distribution
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# In[12]:


#Uses the latentpoints and generator model to create fake images
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


# In[13]:


#Prints the discriminators accuracy for real and fake images
def summary(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    x_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print(f'Epoch {epoch+1} summary:')
    print('Accuracy Real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    #g_model.save(f'more_latent_models/epoch_{epoch+1}')


# In[60]:


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



# In[61]:


#Executes the existing functions
latent_dim = 250
d_model = discriminator_model()
g_model = generator_model(latent_dim)
gan_model = define_gan(g_model, d_model)
train(g_model, d_model, gan_model, normalized_data, latent_dim)


# In[91]:


#Creates and outputs generated images
x_fake_1, y_fake_1 = generate_fake_samples(g_model, latent_dim, 16)
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(x_fake_1[i])
    #plt.imshow((x_fake[i]*255).astype(np.uint8))
    #plt.imshow(x_fake_1[i],cmap="gray_r") should reverse color map, but doesn't
    plt.xticks([])
    plt.yticks([])


# In[93]:


#Generates images using an existing model that was generated on a seperate computer with a GPU

latent_dim2 = 100

loaded_model = tf.keras.models.load_model('gen_models/epoch_70')
x_fake, y_fake = generate_fake_samples(loaded_model, latent_dim2, 16)
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(x_fake[i])
    plt.xticks([])
    plt.yticks([])


# In[ ]:




