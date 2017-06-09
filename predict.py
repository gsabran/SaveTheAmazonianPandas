from tqdm import tqdm
import argparse
import os
import random
from shutil import copyfile
from keras import backend as K
import numpy as np
from keras.models import load_model
from utils import get_uniq_name, get_predictions, get_labels_dict, optimise_f2_thresholds, get_resized_image, get_inputs_shape
from models.simple_cnn import SimpleCNN
from train import TRAINED_MODEL
from constants import LABELS, TRAIN_DATA_DIR, TEST_DATA_DIR
from datasets.dataset import Dataset

def predict(image_files, data_dir, labels=LABELS, thresholds=None, batch_size=64):
	"""
	Yield tuples of predictions as (image_name, probas, tags)
	"""

	print("Starting predictions...")
	imgs = []
	with tqdm(total=len(image_files)) as pbar:
		for f in image_files:
			imgs .append(get_resized_image(f, data_dir, image_data_fmt, input_shape))
			pbar.update(1)
	imgs = np.array(imgs)
	with tqdm(total=len(image_files)) as pbar:
		# larger batch size (relatively to the number of GPU) run out of memory
		proba_predictions = cnn.model.predict(imgs, batch_size=batch_size, verbose=1)
		tag_predictions = get_predictions(proba_predictions, labels, thresholds)
		for f_img, probas, tags in zip(image_files, proba_predictions, tag_predictions):
			yield (f_img, probas, tags)
			pbar.update(1)


if __name__ == "__main__":
	with K.get_session():
		parser = argparse.ArgumentParser(description="test model")
		parser.add_argument("-f", "--file", default="", help="file to test on", type=str)
		parser.add_argument("--data", default="test", help="The set of data to predict on. Either 'test' or 'train'", type=str)
		parser.add_argument("-m", "--model", default=TRAINED_MODEL, help="The model to load", type=str)
		parser.add_argument("-b", "--batch-size", default=64, help="The size of the batch", type=int)
		parser.add_argument("--data-proportion", default=1, help="A proportion of the data to use for training", type=float)
		parser.add_argument("--cpu-only", default=False, help="Wether to only use CPU or not", type=bool)
		args = vars(parser.parse_args())
		print("args", args)

		# it'd be better to load the correct class since implementations
		# of functions such as parallelize might differ
		with open("train/dataset-labels.csv") as f:
			label_idx = np.array([int(i) for i in f.readline().split(",")])
		data = Dataset([], training_files="train/training-files.csv", validation_files="train/validation-files.csv", label_idx=label_idx)
		cnn = SimpleCNN(data, model=load_model(args["model"]), n_gpus=-1 if args["cpu_only"] else 0)

		with open("train/dataset-labels.csv") as f_labels:
			labels = [LABELS[int(i)] for i in f_labels.readline().split(",")]
		print("Predicting for {labels}".format(labels=labels))

		image_data_fmt, input_shape, _ = get_inputs_shape()
		if args["file"] != "":
			img = get_resized_image(args["file"], TRAIN_DATA_DIR, image_data_fmt, input_shape)
			print("Predicting for {fn}".format(fn=args["file"]))
			print(cnn.model.predict(img))
		else:
			"""
			First use validation data to find optimal thresholds to make tag predictions from proba
			"""
			print("Finding optimal thresholds...")
			with open("train/validation-files.csv") as f_validation_files:
				validation_files = f_validation_files.readline().split(",")
				probas = np.array([p for f, p, t in predict(validation_files, TRAIN_DATA_DIR, labels=labels)])
				predicted_labels = get_labels_dict()
				true_tags = [predicted_labels[img] for img in validation_files]
				true_tags = np.array([[x for i, x in enumerate(l) if i in label_idx] for l in true_tags]) # filter to only keep labels of interest
				thresholds = optimise_f2_thresholds(true_tags, probas)
				print("Optimal thresholds are", thresholds)

			data_dir = TRAIN_DATA_DIR if args["data"] == "train" else TEST_DATA_DIR
			
			"""
			Then make tag predictions
			"""
			if args["data"] == "train":
				with open("train/training-files.csv") as f_training_files:
					training_files = f_training_files.readline().split(",")
				list_imgs = training_files + validation_files
			else:
				list_imgs = [f.split(".")[0] for f in sorted(os.listdir(data_dir)) if (".jpg" in f or ".tif" in f)]
				p = args["data_proportion"]
				list_imgs = random.sample(list_imgs, int(len(list_imgs) * args["data_proportion"]))
			
			testId = get_uniq_name()
			predictionFile = "./predict/{d}-predict.csv".format(d=args["data"])
			rawPredictionFile = "./predict/{d}-predict-raw.csv".format(d=args["data"])
			os.makedirs(os.path.dirname(predictionFile), exist_ok=True)

			with open(predictionFile, "w") as pred_f, open(rawPredictionFile, "w") as raw_pred_f:
				pred_f.write("image_name,tags\n")
				raw_pred_f.write("image_name,{tags}\n".format(tags=" ".join(labels)))

				for f_img, probas, tags in predict(list_imgs, data_dir, labels=labels, thresholds=thresholds):
					raw_pred_f.write("{f},{probas}\n".format(f=f_img.split(".")[0], probas=" ".join([str(i) for i in probas])))
					pred_f.write("{f},{tags}\n".format(f=f_img.split(".")[0], tags=" ".join(tags)))

			copyfile(predictionFile, "./predict/archive/{x}-{d}-predict.csv".format(x=testId, d=args["data"]))
			copyfile(rawPredictionFile, "./predict/archive/{x}-{d}-predict-raw.csv".format(x=testId, d=args["data"]))
			print("Done predicting.")
			print("Predictions written to {f}".format(f=predictionFile))
			print("Raw predictions written to {f}".format(f=rawPredictionFile))

