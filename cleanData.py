import os
from shutil import copyfile

# Add more as we encounter them
keywords = ['str', 'ptosis', 'osd']
# new directory to put processed files in: 
newDir = './processedData'
dataDir = './data'

def mkdir(path):
	if not os.path.exists(path):
	    os.makedirs(path)

mkdir(newDir)
for keyword in keywords:
	mkdir(newDir + "/" + keyword)

#remove all videos: 
for (dirpath, dirnames, filenames) in os.walk(dataDir):
	for filename in os.listdir(dirpath):
		if filename.lower().endswith(".mpg") or filename.lower().endswith(".mov"):
			os.remove(dirpath + "/" + filename)


#sort according to disease:
for (dirpath, dirnames, filenames) in os.walk(dataDir):
	pieces = dirpath.lower().split("-")
	for piece in pieces:
		if(piece in keywords):
			for filename in os.listdir(dirpath):
				copyfile(dirpath + '/' + filename, newDir + '/' + piece + '/' + filename)


