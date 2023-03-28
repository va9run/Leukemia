from Model_1.A_Libraries import *
from Model_1.A1_path import *
from Model_1.B_GetData import get_imlist


# Organize images and their labels into a dataframe 
def dataImport(path,pathType):
    imageList = []
    for i,j in zip(pathType,path):
        if i not in ['ALL','HEM']:
            raise ValueError("Entry must be either 'ALL' or 'HEM'")
        else:
            if i == 'HEM':
                loadPath = get_imlist(j) 
                for paths in loadPath:
                    imageList.append({'Images':Image.open(paths),'Labels':0})
            else:
                loadPath = get_imlist(j) 
                for paths in loadPath:
                    imageList.append({'Images':Image.open(paths),'Labels':1})
    imageData = pd.DataFrame(imageList)
    return imageData