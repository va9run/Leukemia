from PIL import Image
import numpy as np
import os


def image_import(path,pathType):
    imageList = []
    labelList = []
    for img, cell_type in zip(path,pathType):
        if cell_type.upper() not in ['ALL','HEM']:
            raise ValueError("Entry must be 'ALL' and 'HEM'")
        else:
            if cell_type.upper() == 'HEM':
                loadPath = [os.path.join(img,f) for f in os.listdir(img) if f.endswith('.bmp')]
                for imgPath in loadPath:
                    imageList.append(Image.open(imgPath))
                    labelList.append(0)
            else:
                loadPath = [os.path.join(img,f) for f in os.listdir(img) if f.endswith('.bmp')]
                for imgPath in loadPath:
                    imageList.append(Image.open(imgPath))
                    labelList.append(1)
    return imageList, labelList