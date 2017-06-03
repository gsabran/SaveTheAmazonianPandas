from shutil import copyfile
import random
import argparse
import os
from keras.models import load_model
from keras import backend as K

from constants import ORIGINAL_DATA_DIR, ORIGINAL_LABEL_FILE
from utils import get_uniq_name
from models.exception import XceptionCNN
from models.simple_cnn import SimpleCNN
from datasets.dataset import Dataset

directory = os.path.dirname(os.path.abspath(__file__))

sessionId = get_uniq_name()
TRAINED_MODEL = "train/model.h5"
os.makedirs("train/archive", exist_ok=True)
os.makedirs("train/tensorboard", exist_ok=True)

if __name__ == "__main__":
	with K.get_session():
		parser = argparse.ArgumentParser(description='train model')
		parser.add_argument('-e', '--epochs', default=10, help='the number of epochs for fitting', type=int)
		parser.add_argument('-b', '--batch-size', default=24, help='the number items per training batch', type=int)
		parser.add_argument('--validation-ratio', default=0.0, help='the proportion of labeled input kept aside of training for validation', type=float)
		parser.add_argument('-g', '--gpu', default=8, help='the number of gpu to use', type=int)
		parser.add_argument('-m', '--model', default='', help='A pre-built model to load', type=str)
		parser.add_argument('-c', '--cnn', default='', help='Which CNN to use. Can be "xception" or left blank for now.', type=str)
		parser.add_argument('--data-proportion', default=1, help='A proportion of the data to use for training', type=float)
		parser.add_argument('--tiff', action='store_true')

		args = vars(parser.parse_args())
		print('args', args)

		N_EPOCH = args['epochs']
		N_GPU = args['gpu']
		BATCH_SIZE = args['batch_size'] * N_GPU
		VALIDATION_RATIO = args['validation_ratio']

		if args['tiff']:
			ORIGINAL_DATA_DIR =  "./rawInput/train-tiff"

		list_imgs = [f.split(".")[0] for f in sorted(os.listdir(ORIGINAL_DATA_DIR))]
		list_imgs = random.sample(list_imgs, int(len(list_imgs) * args['data_proportion']))

		data = Dataset(list_imgs, ORIGINAL_LABEL_FILE, VALIDATION_RATIO, sessionId)
		if args["cnn"] == "xception":
			print("Using Xception architecture")
			cnn = XceptionCNN(data, tiff_model=args['tiff'])
		else:
			print("Using simple model architecture")
			cnn = SimpleCNN(data, tiff_model=args['tiff'])

		if args["model"] != '':
			print("Loading model {m}".format(m=args['model']))
			with open('train/training-files.csv') as f_training_files, open('train/validation-files.csv') as f_validation_files:
				training_files = f_training_files.readline().split(",")
				validation_files = f_training_files.readline().split(",")
				data = Dataset(list_imgs, ORIGINAL_LABEL_FILE, VALIDATION_RATIO, sessionId, training_files=training_files, validation_files=validation_files)
			cnn.model = load_model(args['model'])

		cnn.fit(n_epoch=N_EPOCH, batch_size=BATCH_SIZE, generating=True)
		cnn.model.save(TRAINED_MODEL, overwrite=True)
		copyfile(TRAINED_MODEL, "train/archive/{f}-model.h5".format(f=sessionId))
		print('Done running')