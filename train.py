from shutil import copyfile
import random
import argparse
import os
from keras.models import load_model
from keras import backend as K
import sys

from constants import TRAIN_DATA_DIR
from utils import get_uniq_name, remove
from models.exception import XceptionCNN
from models.vgg16 import VGG16CNN
from models.densenet121 import DenseNet121
from models.ekami_model import AmazonKerasClassifier
from models.simple_cnn import SimpleCNN
from models.gui import GuiNet
from models.parallel_model import get_gpu_max_number
from datasets.dataset import Dataset
from datasets.weather_dataset import WeatherDataset, FilteredDataset
from datasets.weather_in_input import WeatherInInputDataset

directory = os.path.dirname(os.path.abspath(__file__))

sessionId = get_uniq_name()
TRAINED_MODEL = "train/model.h5"



if __name__ == "__main__":
	remove(TRAINED_MODEL)
	remove("train/checkpoint.hdf5")
	remove("train/tensorboard")

	os.makedirs("train/archive", exist_ok=True)
	os.makedirs("train/tensorboard", exist_ok=True)
	MAX_NUMBER_OF_GPUS = get_gpu_max_number()

	with K.get_session():
		parser = argparse.ArgumentParser(description="train model")
		parser.add_argument("-e", "--epochs", default=10, help="the number of epochs for fitting", type=int)
		parser.add_argument("-b", "--batch-size", default=24, help="the number items per training batch", type=int)
		parser.add_argument("--validation-ratio", default=0.1, help="the proportion of labeled input kept aside of training for validation", type=float)
		parser.add_argument("-g", "--gpu", default=MAX_NUMBER_OF_GPUS, help="the number of gpu to use", type=int)
		parser.add_argument("--cpu-only", default=False, help="Wether to only use CPU or not", type=bool)
		parser.add_argument("-m", "--model", default="", help="A pre-built model to load", type=str)
		parser.add_argument("--helper-model", default="", help="A pre-built model to load for further training", type=str)
		parser.add_argument("-c", "--cnn", default="", help='Which CNN to use. Can be "xception", "vgg16" or "ekami" or left blank for now.', type=str)
		parser.add_argument("--data-proportion", default=1, help="A proportion of the data to use for training", type=float)
		parser.add_argument("--generate-data", default=False, help="Wether to generate data or use the original dataset", type=bool)
		parser.add_argument("--dataset", default=None, help="The dataset to use", type=str)
		parser.add_argument("--training-files", default=None, help="Files to use for training", type=str)
		parser.add_argument("--validation-files", default=None, help="Files to use for validation", type=str)
		parser.add_argument("--tta", default=False, help="Wether to use TTA when scoring / predicting", type=bool)

		args = vars(parser.parse_args())
		print("args", args)

		N_EPOCH = args["epochs"]
		if args["cpu_only"]:
			n_gpus = 0
		else:
			n_gpus = args["gpu"]
			if n_gpus == 0:
				print("Error: cannot use 0 GPUs in non CPU only mode")
				sys.exit(1)
			if n_gpus > MAX_NUMBER_OF_GPUS:
				print("Error: only {a} GPUs are available on this machine, while {b} have been requested".format(a=MAX_NUMBER_OF_GPUS, b=n_gpus))
				sys.exit(1)

		BATCH_SIZE = args["batch_size"] * (1 if args["cpu_only"] else n_gpus)
		VALIDATION_RATIO = args["validation_ratio"]

		list_imgs = [f.split(".")[0] for f in sorted(os.listdir(TRAIN_DATA_DIR))]
		list_imgs = random.sample(list_imgs, int(len(list_imgs) * args["data_proportion"]))

		training_files=None
		validation_files=None
		if args["training_files"] is not None:
			with open(args["training_files"]) as f:
				training_files = f.readline().split(",")
		if args["validation_files"] is not None:
			with open(args["validation_files"]) as f:
				validation_files = f.readline().split(",")

		if args["dataset"] == "weather":
			data = WeatherDataset(list_imgs, VALIDATION_RATIO, sessionId=sessionId, training_files=training_files, validation_files=validation_files)
		elif args["dataset"] == "weatherInInput":
			data = WeatherInInputDataset(list_imgs, VALIDATION_RATIO, sessionId=sessionId, training_files=training_files, validation_files=validation_files)
		elif args["dataset"] is not None:
			data = FilteredDataset(list_imgs, args["dataset"], VALIDATION_RATIO, sessionId=sessionId, training_files=training_files, validation_files=validation_files)
		else:
			data = Dataset(list_imgs, VALIDATION_RATIO, sessionId=sessionId, training_files=training_files, validation_files=validation_files)

		if args["cnn"] == "xception":
			print("Using Xception architecture")
			cnn = XceptionCNN(data, n_gpus=n_gpus, with_tta=args["tta"])
		elif args["cnn"] == "vgg16":
			print("Using VGG16 architecture")
			cnn = VGG16CNN(data, n_gpus=n_gpus, with_tta=args["tta"])
		elif args["cnn"] == "ekami":
			print("Using Ekami architecture")
			cnn = AmazonKerasClassifier(data, n_gpus=n_gpus, with_tta=args["tta"])
		elif args["cnn"] == "gui":
			print("Using GuiNet architecture")
			weather_model = load_model(args["helper_model"])
			cnn = GuiNet(weather_model, data, n_gpus=n_gpus, with_tta=args["tta"])
		elif args["cnn"] == "dense121":
			print("Using DenseNet-121 architecture")
			cnn = DenseNet121(data, n_gpus=n_gpus, with_tta=args["tta"])
		else:
			print("Using simple model architecture")
			cnn = SimpleCNN(data, n_gpus=n_gpus, with_tta=args["tta"])

		if args["model"] != "":
			print("Loading model {m}".format(m=args["model"]))
			with open("train/training-files.csv") as f_training_files, open("train/validation-files.csv") as f_validation_files:
				training_files = f_training_files.readline().split(",")
				validation_files = f_validation_files.readline().split(",")
				data = Dataset(list_imgs, VALIDATION_RATIO, sessionId=sessionId, training_files=training_files, validation_files=validation_files)
			cnn.model = load_model(args["model"])

		print("Training for labels {labels}".format(labels=data.labels))
		cnn.fit(n_epoch=N_EPOCH, batch_size=BATCH_SIZE, generating=args["generate_data"])
		cnn.model.save(TRAINED_MODEL, overwrite=True)
		copyfile(TRAINED_MODEL, "train/archive/{f}-model.h5".format(f=sessionId))
		print("Done running")
