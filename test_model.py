import argparse

from model import CNN, TRAINED_MODEL
from skimage.io import imread, imshow, imsave, show

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='test model')
	parser.add_argument('-f', '--file', default='', help='file to test on', type=str)
	args = vars(parser.parse_args())

	cnn = CNN(None)
	cnn.model.load_weights(TRAINED_MODEL)
	if args['file'] != '':
		img = imread(args['file'])
	print(cnn.model.predict(img))

