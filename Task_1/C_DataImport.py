from A_Libraries import *
from A1_path import *
from B_GetData import get_imlist


# Organize images and their labels into a dataframe 
def dataImport(path,pathType):
    imageList = []
    if pathType not in ('ALL','HEM'):
        raise ValueError("Entry must be either 'ALL' or 'HEM'")
    else:
        if pathType == 'HEM':
            loadPath = get_imlist(path) 
            for paths in loadPath:
                imageList.append({'Images':Image.open(paths),'Labels':0})
        else:
            loadPath = get_imlist(path) 
            for paths in loadPath:
                imageList.append({'Images':Image.open(paths),'Labels':1})
    imageData = pd.DataFrame(imageList)
    return imageData