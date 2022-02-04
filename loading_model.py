from cv2 import imread
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import pickle



#loading data
# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()






# #loading trained Network
# new_model = tf.keras.models.load_model('epic_num_reader.model')
# predictions = new_model.predict(x_test)

# #print(predictions)

# detect=5 #passing the input

# print(np.argmax(predictions[detect]))

# plt.imshow(x_test[detect],cmap=plt.cm.binary)
# plt.show()