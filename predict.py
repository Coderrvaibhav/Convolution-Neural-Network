import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("cat_dog_model.h5")

def predict(img_path):

    img = image.load_img(img_path, target_size=(150,150))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0) / 255.0

    pred = model.predict(img_arr)[0][0]

    if pred > 0.5:
        print("Prediction: DOG")
    else:
        print("Prediction: CAT")

# Example
# predict("sample.jpg")
