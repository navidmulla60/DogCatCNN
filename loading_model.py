
import cv2
import tensorflow as tf
import os

DATADIR = "D:\deepLearning\DogCatCNN\PetImages\Dog"
DATADIR2 = "D:\deepLearning\DogCatCNN\PetImages\CAT"
CATEGORIES = ["Dog", "Cat"]

filename="24.jpg"
path_toImage=str(os.path.join(DATADIR,filename))


# img_array = cv2.imread(path_toImage, cv2.IMREAD_GRAYSCALE)
# cv2.imshow("img",img_array)
# cv2.waitKey()
def prepare(filepath):
    IMG_SIZE = 80  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("dogCatTrained.model")

prediction = model.predict([prepare(path_toImage)])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])