import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

import tensorflow as tf

performTrainingData=True

#-----------------------#-----------------------#---------------------#
#location of the data
#-----------------------#-----------------------#---------------------#

DATADIR = "D:\deepLearning\DogCatCNN\PetImages"

CATEGORIES = ["Dog", "Cat"]  # inside PetImages folder we have Two folders called Dog and Cat, Accordingly
                             # we given name in CATEGORIES ARRAY

IMG_SIZE=80 # can be adjustable...

# All images in the training data sholud be of same size, 
# below piece of code will read each images and resize all images 
 
training_data=[]  # create new variable to store resized images

#-----------------------#-----------------------#---------------------#
# creating data
#-----------------------#-----------------------#---------------------#

def createTrainingData():
    
        for category in CATEGORIES:
            
            path=os.path.join(DATADIR,category)
            class_num = CATEGORIES.index(category)
            for img in tqdm(os.listdir(path)):
                try:
                    img_org=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_org, (IMG_SIZE,IMG_SIZE))
                    training_data.append([new_array, class_num])
                #     plt.imshow(new_array,cmap='gray')
                #     plt.show()
                #     break
                # break
                except Exception as e:  # in the interest in keeping the output clean...
                        pass

if performTrainingData:
    createTrainingData()
#print(len(training_data))

#-----------------------#-----------------------#---------------------#
# suffle the training data for better training
#-----------------------#-----------------------#---------------------#

import random

random.shuffle(training_data)

# for sample in training_data[:10]:
#     print(sample[1])

#create two list X for feature y for labels
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

#convert X data to numpy and reshape to image side
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


#-----------------------#-----------------------#---------------------#
#storing the generated data above
#-----------------------#-----------------------#---------------------#

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# load the generated data into training model script.........
