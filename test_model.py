from tqdm import tqdm
import argparse
import os
from shutil import copyfile
from utils import getUniqName
from keras import backend as K

import numpy as np
from model import CNN, TRAINED_MODEL, LABELS
from skimage.io import imread, imshow, imsave, show

DATA_DIR = "./rawInput/test-jpg"

# This will need to go when we retrain as weather will be the first 4 labels
LABELS = sorted(LABELS)
WEATHER_IDX = [5,6,10,11]
WEATHER_VALS = ["clear", "cloudy", "haze", "partly_cloudy"]


def get_pred(y):
	"""
	return the label predictions for an input of shape (n, labelCount)
	"""
	threshold = 0.5
	row_pred = lambda row: [LABELS[k] for k in [WEATHER_IDX[np.argmax(row[WEATHER_IDX])]] + [i for i, v in enumerate(row) if i not in WEATHER_IDX and v > threshold]]
	return (row_pred(row) for row in y)

if __name__ == "__main__":
	with K.get_session():
		parser = argparse.ArgumentParser(description="test model")
		parser.add_argument("-f", "--file", default="", help="file to test on", type=str)
		args = vars(parser.parse_args())

		cnn = CNN(None)
		cnn.model.load_weights(TRAINED_MODEL)

		if args["file"] != "":
			img = imread(args["file"])
			img = img.reshape((1, 256, 256, 3))
			print("Predicting for {fn}".format(fn=args["file"]))
			print(cnn.model.predict(img))
		else:
			print("Reading images...")
			list_imgs = [f for f in os.listdir(DATA_DIR) if (".jpg" in f or ".tif" in f)]
			imgs = []
			with tqdm(total=len(list_imgs)) as pbar:
				for f_img in list_imgs:
					imgs.append(imread(os.path.join(DATA_DIR, f_img)))
					pbar.update(1)
			print("Starting predictions...")
			testId = getUniqName()
			predictionFile = "./test/archive/{x}-predict.csv".format(x=testId)
			rawPredictionFile = "./test/archive/{x}-predict-raw.csv".format(x=testId)
			os.makedirs(os.path.dirname(predictionFile), exist_ok=True)

			with open(predictionFile, "w") as pred_f, open(rawPredictionFile, "w") as raw_pred_f:
				pred_f.write("image_name, tags\n")
				raw_pred_f.write("image_name,{tags}\n".format(tags=", ".join(LABELS)))

				# larger batch size (relatively to the number of GPU) run out of memory
				predictions = cnn.model.predict(np.array(imgs), batch_size=1024, verbose=1)
				for f_img, prediction in zip(list_imgs, predictions):
					raw_pred_f.write("{f}, {tags}\n".format(f=f_img.split(".")[0], tags=", ".join([str(i) for i in prediction])))

				allTags = get_pred(np.array(predictions))
				for f_img, tags in zip(list_imgs, allTags):
					pred_f.write("{f}, {tags}\n".format(f=f_img.split(".")[0], tags=" ".join(tags)))

			copyfile(predictionFile, "./test/predict.csv")
			copyfile(rawPredictionFile, "./test/predict-raw.csv")
			print("Done predicting.")
			print("Predictions written to {f}".format(f=predictionFile))
			print("Raw predictions written to {f}".format(f=rawPredictionFile))

