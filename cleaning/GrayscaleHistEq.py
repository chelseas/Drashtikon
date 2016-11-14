#!/usr/bin/python

import sys
from numpy import *
import modules.histeq as histeq
from PIL import Image, ImageOps

def main():
  imgs = ['test1.jpg', 'test2.jpg', 'test3.jpg']
  for file in imgs:
    im = Image.open(file).convert('L')
    im2 = ImageOps.equalize(im)
    im2.show()
if __name__ == '__main__':
  main()
