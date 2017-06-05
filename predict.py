from tqdm import tqdm
import argparse
import os
import random
from shutil import copyfile
from keras import backend as K
import numpy as np
from skimage.io import imread, imshow, imsave, show
from keras.models import load_model
from constants import LABELS
from utils import get_uniq_name, get_predictions
from models.simple_cnn import SimpleCNN
from train import TRAINED_MODEL
from constants import IMG_ROWS, IMG_COLS, CHANNELS

TEST_DATA_DIR = "./rawInput/test-jpg"
TRAIN_DATA_DIR = "./rawInput/train-jpg"

if __name__ == "__main__":
	with K.get_session():
		parser = argparse.ArgumentParser(description="test model")
		parser.add_argument("-f", "--file", default="", help="file to test on", type=str)
		parser.add_argument("--data", default="test", help="The set of data to predict on. Either 'test' or 'train'", type=str)
		parser.add_argument('-m', '--model', default=TRAINED_MODEL, help='The model to load', type=str)
		parser.add_argument('-b', '--batch-size', default=64, help='The size of the batch', type=int)
		parser.add_argument('--data-proportion', default=1, help='A proportion of the data to use for training', type=float)
		args = vars(parser.parse_args())
		print('args', args)

		cnn = SimpleCNN(multi_gpu=False)
		cnn.model = load_model(args["model"])

		if args["file"] != "":
			img = imread(args["file"])
			img = img.reshape((1, IMG_ROWS, IMG_COLS, CHANNELS))
			print("Predicting for {fn}".format(fn=args["file"]))
			print(cnn.model.predict(img))
		else:
			print("Reading images...")
			data_dir = TRAIN_DATA_DIR if args["data"] == "train" else TEST_DATA_DIR
			
			if args["data"] == "train":
				with open('train/training-files.csv') as f_training_files, open('train/validation-files.csv') as f_validation_files:
					training_files = f_training_files.readline().split(",")
					validation_files = f_validation_files.readline().split(",")
				list_imgs = training_files + validation_files
				list_imgs = ["{f}.jpg".format(f=f) for f in list_imgs]
			else:
				list_imgs = [f for f in sorted(os.listdir(TRAIN_DATA_DIR)) if (".jpg" in f or ".tif" in f)]
				p = args["data_proportion"]
				list_imgs = random.sample(list_imgs, int(len(list_imgs) * args['data_proportion']))
			
			imgs = []
			with tqdm(total=len(list_imgs)) as pbar:
				for f_img in list_imgs:
					imgs.append(imread(os.path.join(data_dir, f_img)))
					pbar.update(1)
			print("Starting predictions...")
			with tqdm(total=len(list_imgs)) as pbar:
				testId = get_uniq_name()
				predictionFile = "./predict/{d}-predict.csv".format(d=args["data"])
				rawPredictionFile = "./predict/{d}-predict-raw.csv".format(d=args["data"])
				os.makedirs(os.path.dirname(predictionFile), exist_ok=True)

				with open(predictionFile, "w") as pred_f, open(rawPredictionFile, "w") as raw_pred_f:
					pred_f.write("image_name,tags\n")
					raw_pred_f.write("image_name,{tags}\n".format(tags=" ".join(LABELS)))

					# larger batch size (relatively to the number of GPU) run out of memory
					predictions = cnn.model.predict(np.array(imgs), batch_size=args["batch_size"], verbose=1)
					for f_img, prediction in zip(list_imgs, predictions):
						raw_pred_f.write("{f},{tags}\n".format(f=f_img.split(".")[0], tags=" ".join([str(i) for i in prediction])))

					allTags = get_predictions(np.array(predictions))
					for f_img, tags in zip(list_imgs, allTags):
						pred_f.write("{f},{tags}\n".format(f=f_img.split(".")[0], tags=" ".join(tags)))
						pbar.update(1)

			copyfile(predictionFile, "./predict/archive/{x}-{d}-predict.csv".format(x=testId, d=args["data"]))
			copyfile(rawPredictionFile, "./predict/archive/{x}-{d}-predict-raw.csv".format(x=testId, d=args["data"]))
			print("Done predicting.")
			print("Predictions written to {f}".format(f=predictionFile))
			print("Raw predictions written to {f}".format(f=rawPredictionFile))

