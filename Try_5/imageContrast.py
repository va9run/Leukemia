from skimage import exposure
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

def img_cont(img):
    img = img_to_array(img)  # Convert the image to a numpy array
    contrasted_img = exposure.equalize_adapthist(img / 255.0, clip_limit=0.15, nbins=256)  # Apply contrast enhancement
    return contrasted_img