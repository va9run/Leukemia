import numpy as np
import tensorflow as tf
from tensorflow.image import resize
import cv2


def crop_img(images, display=False):
    images = np.array(images)
    mask = images == 0

    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    cropped_image = images[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
    
    # Convert cropped_image numpy array to TensorFlow tensor
    resized_image = cv2.resize(cropped_image, (256, 256))

    # Convert resized_image numpy array to TensorFlow tensor
    resized_image_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)
    
    return resized_image_tensor