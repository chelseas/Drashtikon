import numpy
from skimage.feature import hog
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Test of Hog Features.')
parser.add_argument('input' , action="store", help='Input Photo')
args = parser.parse_args()


def main():
  path = os.path.dirname(os.path.realpath(__file__))
  file = os.path.join(path, args.input)
  img = Image.open(file)
  img_r = img.copy()
  img_l = img.copy()
  img.show()
  #img.save("img.png")

  # crop image: https://bytes.com/topic/python/answers/477147-pil-question-about-crop-method
  width = img.size[0]
  newDim = 90
  offsetX = 35
  offsetY = 5
  img_l = img.crop((offsetX, offsetY, offsetX+newDim, newDim + offsetY))
  img_r = img.crop((width-newDim-offsetX, offsetY, width-offsetX, newDim + offsetY))
  img_r.show()
  img_l.show()
  #img_r.save("eye_right.png")
  #img_l.save("eye_left.png")
  assert(img_l.size[0] == img_r.size[0])
  assert(img_l.size[1] == img_r.size[1])
  featuresL, hogImgL = hog(numpy.array(img_l), visualise=True)
  featuresR, hogImgR = hog(numpy.array(img_r), visualise=True)
  Image.fromarray(hogImgL).show()
  Image.fromarray(hogImgR).show()
  #Image.fromarray(hogImgL).save("L_hog.png")
  #Image.fromarray(hogImgR).save("R_hog.png")
  combinedFeatures = numpy.concatenate((featuresL, featuresR))

  # Following code, concatenates cropped images and then extracts HOG.extracts.
  # This is bad because HOG registers a harsh "line" in between the images as an edge.
  # combinedImg = Image.new('L', (newDim*2, newDim))
  # combinedImg.paste(img_l, (0, 0))
  # combinedImg.paste(img_r, (newDim, 0))
  # combinedImg.show()
  # features, hogImg = hog(numpy.array(combinedImg), visualise=True)
  # Image.fromarray(hogImg).show()

  features, regImg = hog(numpy.array(img), visualise=True)
  Image.fromarray(regImg).show()
  Image.fromarray(regImg).save("regImage.png")

  # assert(numpy.array_equal(features,  singularFeatures))

if __name__ == '__main__':
  main()