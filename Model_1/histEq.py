from Model_1.A_Libraries import *

def histeq(im, nbr_bins=255):
    imhist,bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255*cdf/cdf[-1]

    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf