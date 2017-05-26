import numpy as np

from constants import LABELS
from utils import get_predictions

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

def score(rawPredictions, expectations, threshold):
	"""
	Score the predictions for a given threshold
	"""
	predictions = get_predictions(rawPredictions, threshold)
	return np.mean([F2Score(prediction, expectation) for prediction, expectation in zip(predictions, expectations)])

if __name__ == "__main__":
	with open('./train/training-files.csv') as f_train_files:
		train_files = set((int(i) for i in f_train_files.readline().split(",")))

	with open('./predict/train-predict-raw.csv') as f_raw_pred:
		f_raw_pred.readline()
		raw_predictions = {}
		for l in f_raw_pred:
			img_name, values = l.split(",")
			if img_name not in train_files:
				values = [float(v) for v in values.split()]
				raw_predictions[img_name] = values

	with open("./rawInput/train.csv") as f_train:
		expectations = {}
		f_train.readline()
		for l in f_train:
			img_name, tags = l.split(",")
			if img_name not in train_files:
				expectations[img_name] = tags.strip().split()

	validation_idx = [i for i in expectations]

	for threshold in np.linspace(0, 1, 11):
		print("threshold", threshold, "score", score(
			np.array([raw_predictions[i] for i in validation_idx]),
			np.array([expectations[i] for i in validation_idx]),
			threshold
		))
