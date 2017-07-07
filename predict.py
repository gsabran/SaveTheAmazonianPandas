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
from datasets.weather_dataset import WeatherDataset, FilteredDataset
from datasets.weather_in_input import WeatherInInputDataset

def predict(model, image_files, data_dir, image_data_fmt, input_shape, labels=LABELS, input_length=1, thresholds=None, batch_size=64):
	"""
	Yield tuples of predictions as (image_name, probas, tags)
	-Parameter model: the model to use to make predictions
	-Parameter image_files: a list of file names
	-Parameter data_dir: the directory where images are located
	-Parameter labels: the list of labels to predict
	-Parameter input_length: the number of different inputs provided to the model
	-Parameter thresholds: thresholds to use to make predictions
	"""

	print("Starting predictions...")
	inputs = []
	if input_length != 1:
		inputs = [[] for i in range(input_length)]
	with tqdm(total=len(image_files)) as pbar:
		for f in image_files:
			img_input = model.data.get_input(f, data_dir, image_data_fmt, input_shape)
			if input_length != 1:
				for i in range(input_length):
					inputs[i].append(img_input[i])
			else:
				inputs.append(img_input)
			pbar.update(1)

	if input_length != 1:
		for i in range(input_length):
			inputs[i] = np.array(inputs[i])
	else:
		inputs = np.array(inputs)

	with tqdm(total=len(image_files)) as pbar:
		# larger batch size (relatively to the number of GPU) run out of memory
		proba_predictions = model.model.predict(inputs, batch_size=batch_size, verbose=1)
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
		parser.add_argument("--thresholds", default=None, help="A path to a csv representation of the thresholds to use", type=str)
		parser.add_argument("--dataset", default=None, help="The dataset to use", type=str)
		args = vars(parser.parse_args())
		print("args", args)

		with open("train/training-files.csv") as f_training_files, open("train/validation-files.csv") as f_validation_files:
			training_files = f_training_files.readline().split(",")
			validation_files = f_validation_files.readline().split(",")
		

		if args["dataset"] == "weather":
			data = WeatherDataset([], training_files=training_files, validation_files=validation_files)
		elif args["dataset"] == "weatherInInput":
			data = WeatherInInputDataset([], training_files=training_files, validation_files=validation_files)
		elif args["dataset"] != "" and args["dataset"] is not None:
			data = FilteredDataset(list_imgs, args["dataset"], VALIDATION_RATIO, sessionId=sessionId)
		else:
			data = Dataset([], training_files=training_files, validation_files=validation_files)

		labels = data.labels
		label_idx = data.label_idx
		print("Predicting for dataset {ds} with labels {labels}".format(ds=args["dataset"], labels=labels))

		cnn = SimpleCNN(data, model=load_model(args["model"]), n_gpus=-1 if args["cpu_only"] else 0)

		image_data_fmt, input_shape, _ = get_inputs_shape()
		if args["file"] != "":
			img = get_resized_image(args["file"], TRAIN_DATA_DIR, image_data_fmt, input_shape)
			print("Predicting for {fn}".format(fn=args["file"]))
			print(cnn.model.predict(img))
		else:
			"""
			First use validation data to find optimal thresholds to make tag predictions from proba
			"""
			if args["thresholds"] is not None:
				with open(args["thresholds"]) as f:
					thresholds = [float(x) for x in f.readline().strip().split(",")]
				print("Optimal thresholds loaded: {thresholds}".format(thresholds=thresholds))
			else:
				print("Finding optimal thresholds...")
				with open("train/validation-files.csv") as f_validation_files:
					validation_files = f_validation_files.readline().split(",")
					probas = np.array([p for f, p, t in predict(cnn, validation_files, TRAIN_DATA_DIR, image_data_fmt, input_shape, input_length=data.input_length)])
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

				for f_img, probas, tags in predict(cnn, list_imgs, data_dir, image_data_fmt, input_shape, labels=labels, thresholds=thresholds, input_length=data.input_length):
					raw_pred_f.write("{f},{probas}\n".format(f=f_img.split(".")[0], probas=" ".join([str(i) for i in probas])))
					pred_f.write("{f},{tags}\n".format(f=f_img.split(".")[0], tags=" ".join(tags)))

			copyfile(predictionFile, "./predict/archive/{x}-{d}-predict.csv".format(x=testId, d=args["data"]))
			copyfile(rawPredictionFile, "./predict/archive/{x}-{d}-predict-raw.csv".format(x=testId, d=args["data"]))
			print("Done predicting.")
			print("Predictions written to {f}".format(f=predictionFile))
			print("Raw predictions written to {f}".format(f=rawPredictionFile))

