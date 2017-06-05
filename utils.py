from __future__ import division
from shutil import rmtree
import subprocess
import time
import string
import numpy as np
import os
import math

from bisect import bisect
from random import random
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

def files_proba(file_labels, labels):
	# file_labels is a dict of (img -> binary vector of labels)
	train_tags = file_labels

	count_labels = np.array([0] * len(labels))
	for img in train_tags:
		count_labels += train_tags[img]
	n_doc = len(train_tags)
	# idf is a vector representing the idf of each tag
	idf = np.array([math.log(n_doc / (1 + count_labels[i])) for i, _ in enumerate(labels)])

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
