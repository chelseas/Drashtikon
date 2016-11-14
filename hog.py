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
  img.show()
  features, hogImg = hog(numpy.array(img), visualise=True)
  Image.fromarray(hogImg).show()
  print(type(features))
  print(features.size)
  print(features[0])
  print(features)
  


if __name__ == '__main__':
  main()