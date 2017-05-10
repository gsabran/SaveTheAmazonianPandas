import subprocess
import time
import string
import numpy as np
import os

from constants import LABELS, WEATHER_IDX, DATA_DIR

def get_uniq_name():
	"""
	A uniq name to diffenrenciate generated files
	"""
	gitHash = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
	gitMessage =  subprocess.check_output(["git", "log", "--format=%B", "-n", "1", "HEAD"]).strip().decode("utf-8")
	# remove bad filename characters: http://stackoverflow.com/a/295146/2054629
	valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
	gitMessage = ''.join(c for c in gitMessage if c in valid_chars)

	t = time.strftime("%Y-%m-%d.%H:%M:%S")
	return "{t}-{h}-{m}".format(t=t, h=gitHash, m=gitMessage).lower().replace(" ", "_")


def get_predictions(y, threshold=0.5):
	"""
	return the label predictions for an input of shape (n, labelCount)
	"""
	row_pred = lambda row: [LABELS[k] for k in [WEATHER_IDX[np.argmax(row[WEATHER_IDX])]] + [i for i, v in enumerate(row) if i not in WEATHER_IDX and v > threshold]]
	return (row_pred(row) for row in y)

def get_generated_images(originalImageFileName, ext="jpg"):
	"""
	return an array of images generated from an original image.
	originalImageFileName should not contain extension nor directory path
	"""
	for i in range(8):
		newFileName = "{n}--{i}.{ext}".format(n=originalImageFileName, i=i, ext=ext)
		yield os.path.join(DATA_DIR, newFileName)
