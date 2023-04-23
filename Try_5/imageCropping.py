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
    
    # Resize cropped images
    resized_image = cv2.resize(cropped_image, (256, 256))

    resized_image = resized_image.astype('float32')/255.0
    return resized_image