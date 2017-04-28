from tqdm import tqdm
import argparse
import os
from shutil import copyfile
from keras import backend as K
import numpy as np
from skimage.io import imread, imshow, imsave, show
from keras.models import load_model
from constants import LABELS
from utils import get_uniq_name, get_predictions
from model import CNN, TRAINED_MODEL, IMG_ROWS, IMG_COLS, CHANNELS, ModelCNN

TEST_DATA_DIR = "./rawInput/test-jpg"
TRAIN_DATA_DIR = "./rawInput/train-jpg"

if __name__ == "__main__":
	with K.get_session():
		parser = argparse.ArgumentParser(description="test model")
		parser.add_argument("-f", "--file", default="", help="file to test on", type=str)
		parser.add_argument("--data", default="test", help="The set of data to predict on. Either 'test' or 'train'", type=str)
		parser.add_argument('-m', '--model', default=TRAINED_MODEL, help='The model to load', type=str)
		args = vars(parser.parse_args())
		print('args', args)

		cnn = ModelCNN()
		cnn.model = load_model(args["model"])

		if args["file"] != "":
			img = imread(args["file"])
			img = img.reshape((1, IMG_ROWS, IMG_COLS, CHANNELS))
			print("Predicting for {fn}".format(fn=args["file"]))
			print(cnn.model.predict(img))
		else:
			print("Reading images...")
			data_dir = TRAIN_DATA_DIR if args["data"] == "train" else TEST_DATA_DIR

			list_imgs = [f for f in sorted(os.listdir(data_dir)) if (".jpg" in f or ".tif" in f)]
			imgs = []
			with tqdm(total=len(list_imgs)) as pbar:
				for f_img in list_imgs:
					imgs.append(imread(os.path.join(data_dir, f_img)))
					pbar.update(1)
			print("Starting predictions...")
			testId = get_uniq_name()
			predictionFile = "./predict/{d}-predict.csv".format(d=args["data"])
			rawPredictionFile = "./predict/{d}-predict-raw.csv".format(d=args["data"])
			os.makedirs(os.path.dirname(predictionFile), exist_ok=True)

			with open(predictionFile, "w") as pred_f, open(rawPredictionFile, "w") as raw_pred_f:
				pred_f.write("image_name, tags\n")
				raw_pred_f.write("image_name, {tags}\n".format(tags=" ".join(LABELS)))

				# larger batch size (relatively to the number of GPU) run out of memory
				predictions = cnn.model.predict(np.array(imgs), batch_size=1024, verbose=1)
				for f_img, prediction in zip(list_imgs, predictions):
					raw_pred_f.write("{f}, {tags}\n".format(f=f_img.split(".")[0], tags=" ".join([str(i) for i in prediction])))

				allTags = get_predictions(np.array(predictions))
				for f_img, tags in zip(list_imgs, allTags):
					pred_f.write("{f}, {tags}\n".format(f=f_img.split(".")[0], tags=" ".join(tags)))

			copyfile(predictionFile, "./predict/archive/{x}-{d}-predict.csv".format(x=testId, d=args["data"]))
			copyfile(rawPredictionFile, "./predict/archive/{x}-{d}-predict-raw.csv".format(x=testId, d=args["data"]))
			print("Done predicting.")
			print("Predictions written to {f}".format(f=predictionFile))
			print("Raw predictions written to {f}".format(f=rawPredictionFile))

