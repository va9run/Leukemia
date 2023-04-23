import numpy as np
import tensorflow
from tensorflow.image import resize


def crop_img(images, display=False):
    mask = images == 0

    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    cropped_image = images[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
    cropped_image = resize(cropped_image,[256,256])
    return cropped_image