#!/usr/bin/python

import os
import numpy
from PIL import Image, ImageOps
import openface
import argparse
from datetime import datetime

path = os.path.dirname(os.path.realpath(__file__))

#From /demo in openface package
fileDir = os.path.dirname(os.path.realpath(openface.__file__))
modelDir = os.path.join(fileDir, 'models') 
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

alignDlibArgs = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
imgDim = 300
cuda = True
align = openface.AlignDlib(alignDlibArgs)

parser = argparse.ArgumentParser(description='Face identification and cropping script.')
parser.add_argument('inputDir' , action="store", help='Relative path to the images directory')
parser.add_argument('--test' , action="store_true", help='Only crops 20 images.')
#parser.add_argument('--normalize' , action="store_true", help='Intensity normaliztion.')
args = parser.parse_args()

def cropOpenFace(file, outputDirectory):
  img =  numpy.array(ImageOps.equalize(Image.open(file).convert('L')))
  bb = align.getLargestFaceBoundingBox(img) 
  alignedFace = align.align(imgDim, img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
  if alignedFace is None:
    #print("No face found with OUTER_EYES_AND_NOSE: ", file)
    alignedFace = align.align(imgDim, img, bb, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    #if alignedFace is None:
      #print("No face found with INNER_EYES_AND_BOTTOM_LIP: ", file)
  name = os.path.basename(os.path.splitext(file)[0])
  if alignedFace is None:
      img = Image.fromarray(img)
      name = os.path.join(outputDirectory, 'uncropped',  name + '.jpg')
      img.save(name)
  else: 
      cropped = Image.fromarray(alignedFace)
      cropped = cropped.crop((0, 0, 300, 150))
      name = os.path.join(outputDirectory, 'cropped', name + '.jpg')
      cropped.save(name)

def main():
  photoDirectory = os.path.join(path, args.inputDir)
  outputDirectory = os.path.join(path, args.inputDir, '..', os.path.basename(args.inputDir)+ '_output_' + datetime.now().strftime("%m%d%H%M%S"))
  os.mkdir(outputDirectory)
  os.mkdir(os.path.join(outputDirectory, 'cropped'))
  os.mkdir(os.path.join(outputDirectory,  'uncropped'))
  imgFiles = [f for f in os.listdir(photoDirectory) if (os.path.isfile(os.path.join(photoDirectory, f)) and any(f.endswith(ext) for ext in ['.jpg', '.JPG', '.png']))]
  if args.test:
    count = 0
    for img in imgFiles:
      cropOpenFace(os.path.join(photoDirectory, img), outputDirectory)
      count += 1
      if count == 20:
        return
  else: 
    counter = 0
    for img in imgFiles:
      if (counter % 100 == 0): print("Cropping Directory: {} -- Completed {} of {} total files.").format(args.inputDir, counter, len(imgFiles))
      cropOpenFace(os.path.join(photoDirectory, img), outputDirectory)
      counter += 1



if __name__ == '__main__':
  main()
