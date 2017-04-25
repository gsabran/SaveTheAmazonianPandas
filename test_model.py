from tqdm import tqdm
import argparse
import os
from utils import getUniqName
from keras import backend as K

# This forces predicting on CPUs instead of GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
			predictionFile = "./test/{x}-predict.csv".format(x=getUniqName())
			with open(predictionFile, "w") as pred_f:
				pred_f.write("image_name,tags\n")
				prediction = cnn.model.predict(np.array(imgs), batch_size=128, verbose=1)
				allTags = get_pred(np.array(prediction))
				for f_img, tags in zip(list_imgs, allTags):
					pred_f.write("{f}, {tags}\n".format(f=f_img.split(".")[0], tags=" ".join(tags)))
			print("Done predicting. Predictions written to {f}".format(f=predictionFile))
