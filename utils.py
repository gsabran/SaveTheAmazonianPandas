from __future__ import division
from shutil import rmtree
import subprocess
import time
import string
import numpy as np
from sklearn.metrics import fbeta_score
from skimage.io import imread
from scipy.misc import imresize
import os
import math
from keras import backend as K

from bisect import bisect
from random import random
from constants import LABELS, WEATHER_IDX, DATA_DIR, ORIGINAL_LABEL_FILE, CHANNELS, IMG_ROWS, IMG_COLS, IMG_SCALE, TRAIN_DATA_DIR

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


def get_predictions(y, thresholds):
	"""
	return the label predictions for an input of shape (n, labelCount)
	"""
	row_pred = lambda row: [LABELS[k] for k in [WEATHER_IDX[np.argmax(row[WEATHER_IDX])]] + [i for i, v in enumerate(row) if i not in WEATHER_IDX and v > thresholds[i]]]
	return (row_pred(row) for row in y)

# from https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
	"""
	find optimal thresholds to predict labels
	- Parameter y: a (n) array of 0 - 1 representing true values
	- Parameter p: a (n, 17) matrix of probabilies
	"""
	def mf(x):
		p2 = np.zeros_like(p)
		for i in range(len(LABELS)):
			p2[:, i] = (p[:, i] > x[i]).astype(np.int)
		score = fbeta_score(y, p2, beta=2, average='samples')
		return score

	x = [0.2] * len(LABELS)
	for i in range(len(LABELS)):
		best_i2 = 0
		best_score = 0
		for i2 in range(resolution):
			i2 /= resolution
			x[i] = i2
			score = mf(x)
			if score > best_score:
				best_i2 = i2
				best_score = score
		x[i] = best_i2

	return x

def get_labels_dict():
	"""
	Return a dictionary of { image_name: [0-1 tag] }
	"""
	labels_dict = {}
	with open(ORIGINAL_LABEL_FILE) as f:
		f.readline()
		for l in f:
			filename, rawTags = l.strip().split(',')
			tags = rawTags.split(' ')
			bool_tags = [1 if tag in tags else 0 for tag in LABELS]
			file = filename.split('/')[-1].split('.')[0]
			labels_dict[file] = bool_tags
	return labels_dict

def get_inputs_shape():
	"""
	Return the image format, and the share images should have
	"""
	image_data_fmt = K.image_data_format()
	if image_data_fmt == 'channels_first':
		input_shape = (CHANNELS, int(IMG_ROWS * IMG_SCALE), int(IMG_COLS * IMG_SCALE))
	else:
		input_shape = (int(IMG_ROWS * IMG_SCALE), int(IMG_COLS * IMG_SCALE), CHANNELS)
	return image_data_fmt, input_shape

def get_resized_image(f, data_dir, image_data_fmt, input_shape):
	"""
	Read the image from file and return it in the expected size
	"""
	img = imread(os.path.join(data_dir, "{}.jpg".format(f)))
	if image_data_fmt == 'channels_first':
		img = img.reshape((CHANNELS, IMG_ROWS, IMG_COLS))
	return imresize(img, input_shape)

# deprecated
def get_generated_images(originalImageFileName, ext="jpg"):
	"""
	return an array of images generated from an original image.
	originalImageFileName should not contain extension nor directory path
	"""
	for i in range(8):
		newFileName = "{n}--{i}.{ext}".format(n=originalImageFileName, i=i, ext=ext)
		yield os.path.join(DATA_DIR, newFileName)

def files_proba(file_labels, labels):
	# file_labels is a dict of (img -> binary vector of labels)
	train_tags = file_labels

	count_labels = np.array([0] * len(labels))
	for img in train_tags:
		count_labels += train_tags[img]
	n_doc = len(train_tags)
	# idf is a vector representing the idf of each tag
	# We've modified this idf by removing the log here, which leads
	# to more balanced sampling
	idf = np.array([n_doc / (1 + count_labels[i]) for i, _ in enumerate(labels)])

	tf_idf = {}
	for img in train_tags:
		# We use train_tags[img] as a mask over the idf of each tag and sum
		tf_idf[img] = (np.array(train_tags[img]) * idf).sum()

	sum_tf = sum(tf_idf.values())
	proba = {img: tf_idf[img] / sum_tf for img in tf_idf}
	return proba

def files_and_cdf_from_proba(proba):
	files_probs = sorted(proba.items(), key=lambda i: i[1])
	return list(map(lambda i: i[0], files_probs)), np.cumsum(list(map(lambda i: i[1], files_probs)))

def pick(n, files, cdf):
	return [files[bisect(cdf, random())] for i in range(n)]

def remove(path):
	"""
	Remove a file or directory if it exists, else do nothing
	"""
	try:
		if os.path.isdir(path):
			rmtree(path)
		else:
			os.remove(path)
	except FileNotFoundError:
		pass

def F2Score(predicted, actual):
	# see https://www.kaggle.com/c/planet-understanding-the-amazon-from-space#evaluation
	predicted = set(predicted)
	actual = set(actual)
	tp = len(predicted & actual)
	tn = len(LABELS) - len(predicted | actual)
	fp = len(predicted) - tp
	fn = (len(LABELS) - len(predicted)) - tn
	p = tp / (tp + fp)
	r = tp / (tp + fn)
	if p == 0 or r == 0:
		return 0
	b = 2
	return (1 + b**2) * p * r / (b**2*p + r)

