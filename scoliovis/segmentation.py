# Pseudo-script for segmentation.
# sunhl-1th-01-Mar-2017-310 a ap
import base64
import cv2 as cv
# from fastapi import UploadFile
from mat4py import loadmat
from PIL import Image
import numpy as np

# from keras.models import load_model
# import numpy as np
import tensorflow as tf
# from draw_landmarks import draw_landmarks

# == PSEUDO SEGMENT ==

async def segment(image: Image):
    # Convert PIL.Image -> Image (CV2)
    image = np.asarray(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    data = loadmat(
        "data/sunhl-1th-01-Mar-2017-310 a ap.jpg.mat")

    # data contains a dict with an only 'p2' key value, so we get the first value (print("data") if confused)
    data = data['p2']
    for x, y in data:
        cv.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=20)
    
    # Encode to JPEG
    _, encoded_img = cv.imencode('.JPG', image)

    # Get base64
    encoded_img = base64.b64encode(encoded_img)
    
    return encoded_img

# == MODEL SEGMENT ==
from scoliovis.get_model import get_model
from scoliovis.draw_landmarks import draw_landmarks

model = get_model()

async def segment_with_model(image: Image):
    # Convert PIL.Image -> ImageData (TF)
    image = tf.keras.utils.img_to_array(image)
    h, w = image.shape[0], image.shape[1]

    # Preprocessing for Model
    img_input = tf.cast(image, tf.float32) / 255.0  # normalize
    img_input = tf.image.resize(img_input, [256, 128])
    img_input = tf.expand_dims(img_input, 0)

    # Predict Model
    prediction = model.predict(img_input)

    # Generate Image Landmarks
    image = draw_landmarks(image, h, w, prediction[0])

    # Encode to JPEG
    _, encoded_img = cv.imencode('.JPG', image)

    # Get base64
    encoded_img = base64.b64encode(encoded_img)

    return encoded_img

# async def segment(img: Image.Image):
#     # # Preprocess To Model Input
#     img = tf.keras.utils.img_to_array(img)
#     height = img.shape[0]
#     width = img.shape[1]

#     img_input = tf.cast(img, tf.float32) / 255.0  # normalize
#     img_input = tf.image.resize(img_input, [256, 128])
#     img_input = tf.expand_dims(img_input, 0)

#     # # Predict with Model
#     prediction = model.predict(img_input)

#     # # Generate Image with Landmarks
#     return draw_landmarks(img, height, width, prediction[0])
