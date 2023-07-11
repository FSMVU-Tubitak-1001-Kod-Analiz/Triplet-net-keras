#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import random
import matplotlib.patheffects as PathEffects


# In[2]:


from keras.layers import Input, Conv2D, Conv1D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
# from tensorflow.keras.optimizers import SGD
from keras.losses import binary_crossentropy
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


# In[3]:


from itertools import permutations


# In[4]:


import seaborn as sns


# In[5]:


from keras.datasets import mnist
from sklearn.manifold import TSNE


# In[6]:


from sklearn.svm import SVC
import pandas as pd
import tensorflow as tf


# In[7]:


#(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[7]:





# In[8]:


def get_train_test_dataset(train_test_ratio):
    X = []
    y = []
    X = pd.read_excel(io='CompareResult4.xlsx', sheet_name='Sheet1',usecols="J,K,L")
    X.fillna(0,inplace=True)
    X=X.to_numpy()
    return X


# In[9]:



x_train = get_train_test_dataset(0.5)



#x_train=tf.convert_to_tensor(x_train) 


# In[10]:


# Define our own plot function
def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.savefig(subtitle)


# In[11]:


data_size=8396
#x_train_flat_anchor = tf.reshape(x_train[0:,0], [data_size, 1])

x_train_flat_anchor = x_train[0:,0]
x_train_flat_positive = x_train[0:,1]
x_train_flat_negative = x_train[0:,2]

#print(x_train_flat_anchor)
#x_test_flat = x_test.reshape(-1,784)


# In[12]:


#tsne = TSNE()
#train_tsne_embeds = tsne.fit_transform(x_train_flat[:512])
#scatter(train_tsne_embeds, y_train[:512], "Samples from Training Data")

#eval_tsne_embeds = tsne.fit_transform(x_test_flat[:512])
#scatter(eval_tsne_embeds, y_test[:512], "Samples from Validation Data")


# In[13]:


Classifier_input = Input((784,))
Classifier_output = Dense(10, activation='softmax')(Classifier_input)
Classifier_model = Model(Classifier_input, Classifier_output)


# In[14]:


from sklearn.preprocessing import LabelBinarizer


# In[15]:


le = LabelBinarizer()


# In[16]:


#y_train_onehot = le.fit_transform(y_train)
#y_test_onehot = le.transform(y_test)


# In[17]:


Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[18]:


#Classifier_model.fit(x_train_flat,y_train_onehot, validation_data=(x_test_flat,y_test_onehot),epochs=10)


# In[19]:


def generate_triplet():
    x= get_train_test_dataset(0.7);  
                
    return x;


# In[20]:


X_train = generate_triplet()


# In[21]:


print(X_train.shape)


# ## Triplet NN

# In[21]:





# In[22]:


def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    #model.add(Input(tensor=tf_embedding_input))
    #model.add(Flatten(input_shape=(1,1,1,))) #this line
    #model.add(Conv1D(1,(7,7),padding='same',input_shape=(in_dims[0],in_dims[1],in_dims[2],),activation='relu',name='conv1'))
    #model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool1'))
    #model.add(Conv1D(10,(1),padding='same',activation='relu',name='conv2'))
    #model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool2'))
    #model.add(Flatten(name='flatten'))
    #model.add(Dense(10,name='embeddings'))
    # model.add(Dense(600))

    return model


# In[23]:


adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)


# In[24]:


data_size=1
anchor_input = Input((1, ), name='anchor_input')
positive_input = Input((1, ), name='positive_input')
negative_input = Input((1, ), name='negative_input')

# Shared embedding layer for positive and negative items
Shared_DNN = create_base_network([1,])


encoded_anchor = Shared_DNN(anchor_input)
encoded_positive = Shared_DNN(positive_input)
encoded_negative = Shared_DNN(negative_input)


merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
print(merged_vector.shape)
model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)



# In[25]:


#from tensorflow import keras
#loss_fn = keras.losses.SparseCategoricalCrossentropy()
#model.compile(loss=loss_fn, optimizer=adam_optim)


# In[26]:


model.summary()


# In[27]:


Anchor = x_train[:,0]
Positive = x_train[:,1]
Negative = x_train[:,2]
Anchor_test = x_train[:,0]
Positive_test = x_train[:,1]
Negative_test = x_train[:,2]

Y_dummy = np.empty((Anchor.shape[0],1))
print(Y_dummy.size)
Y_dummy2 = np.empty((Anchor_test.shape[0],1))
print(Y_dummy2.size)


# In[28]:


[Anchor,Positive,Negative]


# In[35]:


def triplet_loss(y_true, y_pred, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    import sys
    print('y_pred.shape = ',y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12


    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]


    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    print('pos_dist = ',pos_dist)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    print('neg_dist = ',neg_dist)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    print('basic_loss = ',basic_loss)

    """
        basic_loss = pos_dist-neg_dist+alpha
        Node: 'triplet_loss/add'
        Incompatible shapes: [10,3] vs. [10]
	    [[{{node triplet_loss/add}}]] [Op:__inference_train_function_364]
    """

    loss = K.maximum(basic_loss,0.0)

    #return tf.constant(loss.data[0])
    #return tf.constant(0)
    return loss

model.compile(loss=triplet_loss, optimizer=adam_optim)


model.fit([Anchor,Positive,Negative],y=Y_dummy,validation_data=([Anchor_test,Positive_test,Negative_test],Y_dummy2), batch_size=10, epochs=10)


# In[36]:


trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)


# In[38]:


# trained_model.load_weights('triplet_model_MNIST.hdf5')


# In[41]:


tsne = TSNE()
X_train_trm = trained_model.predict(x_train[:512].reshape(-1,1))
# X_test_trm = trained_model.predict(x_test[:512].reshape(-1,28,28,1))
train_tsne_embeds = tsne.fit_transform(X_train_trm)
# eval_tsne_embeds = tsne.fit_transform(X_test_trm)


# In[42]:


train_tsne_embeds


# In[ ]:


# scatter(train_tsne_embeds, y_train[:512], "Training Data After TNN")


# In[ ]:


# scatter(eval_tsne_embeds, y_test[:512], "Validation Data After TNN")


# In[ ]:


# X_train_trm = trained_model.predict(x_train.reshape(-1,28,28,1))
# X_test_trm = trained_model.predict(x_test.reshape(-1,28,28,1))
#
# Classifier_input = Input((4,))
# Classifier_output = Dense(10, activation='softmax')(Classifier_input)
# Classifier_model = Model(Classifier_input, Classifier_output)
#
#
# Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# Classifier_model.fit(X_train_trm,y_train_onehot, validation_data=(X_test_trm,y_test_onehot),epochs=10)


# In[ ]:




