import os
import constants
import numpy as np
from scipy import misc, ndimage

def resize(image, dim1, dim2):
	return misc.imresize(image, (dim1, dim2))

def fileWalk(directory, destPath):
	try: 
		os.makedirs(destPath)
	except OSError:
		if not os.path.isdir(destPath):
			raise

	for subdir, dirs, files in os.walk(directory):
		for file in files:
			if len(file) <= 4 or file[-4:] != '.jpg':
				continue

			pic = misc.imread(os.path.join(subdir, file))
			dim1 = len(pic)
			dim2 = len(pic[0])
			if dim1 > dim2:
				pic = np.rot90(pic)

			picResized = resize(pic,constants.DIM1, constants.DIM2)
			misc.imsave(os.path.join(destPath, file), picResized)
		

def main():
	prepath = os.path.join(os.getcwd(), 'dataset-original')
	glassDir = os.path.join(prepath, 'glass')
	paperDir = os.path.join(prepath, 'paper')
	cardboardDir = os.path.join(prepath, 'cardboard')
	plasticDir = os.path.join(prepath, 'plastic')
	metalDir = os.path.join(prepath, 'metal')
	trashDir = os.path.join(prepath, 'trash')

	destPath = os.path.join(os.getcwd(), 'dataset-resized')
	try: 
		os.makedirs(destPath)
	except OSError:
		if not os.path.isdir(destPath):
			raise

	#GLASS
	fileWalk(glassDir, os.path.join(destPath, 'glass'))

	#PAPER
	fileWalk(paperDir, os.path.join(destPath, 'paper'))

	#CARDBOARD
	fileWalk(cardboardDir, os.path.join(destPath, 'cardboard'))

	#PLASTIC
	fileWalk(plasticDir, os.path.join(destPath, 'plastic'))

	#METAL
	fileWalk(metalDir, os.path.join(destPath, 'metal'))

	#TRASH
	fileWalk(trashDir, os.path.join(destPath, 'trash'))  

if __name__ == '__main__':
    main()