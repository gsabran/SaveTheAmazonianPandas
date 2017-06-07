from __future__ import division
from shutil import rmtree
import subprocess
import time
import string
import numpy as np
from sklearn.metrics import fbeta_score
import os
import math

from bisect import bisect
from random import random
from constants import LABELS, WEATHER_IDX, DATA_DIR, ORIGINAL_LABEL_FILE

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
		if verbose:
			print(i, best_i2, best_score)

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
			bool_tags = [1 if tag in tags else 0 for tag in self.labels]
			file = filename.split('/')[-1].split('.')[0]
			labels_dict[file] = bool_tags
	return labels_dict

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
