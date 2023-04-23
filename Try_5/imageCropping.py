import numpy as np
import tensorflow as tf
from tensorflow.image import resize


def crop_img(images, display=False):
    images = np.array(images)
    mask = images == 0

    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    cropped_image = images[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
    
    # Convert cropped_image numpy array to TensorFlow tensor
    cropped_image_tensor = tf.convert_to_tensor(cropped_image, dtype=tf.float32)
    resized_image = resize(cropped_image_tensor,[256,256])
    return resized_image