from numpy import *
from PIL import Image


def histeq(im,nbr_bins=256):
   #get image histogram
   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)
   print(im2.shape)

   return Image.fromarray(im2.reshape(im.shape), 'L'), cdf