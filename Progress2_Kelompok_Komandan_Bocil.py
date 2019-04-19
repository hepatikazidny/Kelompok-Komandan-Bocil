#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


print ("=========================")
print ("Global Feature Extraction")
print ("=========================")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import pywt

# fixed-sizes for image
# fixed_size = tuple((500, 500))
fixed_size = (250, 250)

# path to training data
train_path = os.path.abspath("cat/train/")

# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.20

# seed for reproducing same results
seed = 9


# In[2]:


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# In[3]:


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# In[4]:


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# In[5]:


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
wavelet_features=[]
labels = []

i, j = 0, 0
k = 0
t=0

# num of images per class
images_per_class = 80


# In[6]:


# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name
    
    print(current_label)
    
    list_images = [f for f in glob.glob(dir + "\*.jpg")]
    list_images = sorted(list_images)
#     print("list image ",list_images[0])
    

    k = 11
    # loop over the images in each sub-folder
#     for x in range(1,images_per_class+1):
    for x in range(1,images_per_class+1):
        # get the image file name
#         file = dir + "\\" + current_label + "_" +str(x) + ".jpg"
        file = list_images[x]
#         print(file)
#         print(os.path.isfile(file))

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
#         image = cv2.
#         print(image.shape)
#         cv2.imshow("Test", image)
        image = cv2.resize(image, fixed_size)
#         print(image)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
#         print (fv_hu_moments)

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)
#         print(global_feature)

        i += 1
        k += 1
    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1
#     fig.tight_layout()
#     plt.imshow(image)
#     plt.show()
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))
print ("[STATUS] completed Global Feature Extraction...")


# In[9]:



for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name
    
    print(current_label)
    
    list_images = [f for f in glob.glob(dir + "\*.jpg")]
    list_images = sorted(list_images)
    t = 11
    wavelet=[]   
#     wavelet_feature=[]
    for x in range(1,images_per_class+1):
        file = list_images[x]
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        gray_images=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
        
        coeffs2 = pywt.dwt2(gray_images[:,:], 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        wavelet.append(coeffs2)
        wavelet_array=np.array(wavelet) 
#         print(wavelet_array)
        
        fig = plt.figure(figsize=(12,3))
        for i, a in enumerate([LL,LH,HL,HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        i += 1
        t += 1
#         wavelet_array.append(wavelet)
#         wavelet_feature = np.array(wavelet_array)
        
        wavelet_features=np.expand_dims(wavelet_array,axis=1)
#         wavelets.append(wavelet_features)
        
    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1
    
#     plt.imshow(image)
    fig.tight_layout()
    plt.show()
print ("[STATUS] feature vector size {}".format(np.array(wavelet_array).shape))
print ("[STATUS] feature vector size {}".format(np.array(wavelet_features).shape))


# In[ ]:




