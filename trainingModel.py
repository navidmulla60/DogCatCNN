

import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
#Load created Data trained data
# Note: X for feature y for labels 

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

#normalize the data (0-1)

# X = X/255.0
X=tf.keras.utils.normalize(X, axis=1)
X=np.array(X)
y=np.array(y)
shape=X.shape[1:]
print(shape)


#now create a model
# model = Sequential()
model = Sequential()
model.add(Conv2D(15, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(15, (3, 3)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(20))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# Compiling model

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("------------------------------------------------")

# #train the model

model.fit(X, y, batch_size=50, epochs=10, validation_split=0.1)




#saving the model
model.save('dogCatTrained.model')


