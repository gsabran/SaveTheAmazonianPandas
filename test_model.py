import argparse
import os
import numpy as np
from model import CNN, TRAINED_MODEL, LABELS
from skimage.io import imread, imshow, imsave, show

DATA_DIR = "./rawInput/test-jpg"

# This will need to go when we retrain as weather will be the first 4 labels
LABELS = sorted(LABELS)
WEATHER_IDX = [5,6,10,11]
WEATHER_VALS = ["clear", "cloudy", "haze", "partly_cloudy"]

def get_pred(y_hat):
	weather = WEATHER_VALS[np.argmax(y_hat[WEATHER_IDX])]
	other_tags = [LABELS[i] for i in np.where(y_hat > 0.5)[0] if i not in WEATHER_IDX]
	return (weather, other_tags)

if __name__ == "__main__":
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
		list_imgs = [f for f in os.listdir(DATA_DIR) if (".jpg" in f or ".tif" in f)]
		with open("./test/predict.csv", "w") as pred_f:
			pred_f.write("image_name,tags\n")
			for f_img in list_imgs:
				img = imread(os.path.join(DATA_DIR, f_img))
				img = img.reshape((1, 256, 256, 3))
				y_hat = np.array(cnn.model.predict(img)).reshape((len(LABELS),))
				weather, other_tags = get_pred(y_hat)
				pred_f.write("{f},{w} {other}\n".format(f=f_img.split(".")[0], w=weather, other=" ".join(other_tags)))
