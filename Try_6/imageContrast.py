from skimage import exposure
import numpy as np

def img_cont(img):
    img = np.array(img)  # Convert the image to a numpy array
    contrasted_img = exposure.equalize_adapthist(img / 255.0, clip_limit=0.15, nbins=256)  # Apply contrast enhancement
    return contrasted_img