from model import LABELS
from test_model import get_pred
import numpy as np

# This will need to go when we retrain as weather will be the first 4 labels
LABELS = sorted(LABELS)

def F2Score(predicted, actual):
  print("F2Score", predicted, actual)
  # see https://www.kaggle.com/c/planet-understanding-the-amazon-from-space#evaluation
  predicted = set(predicted)
  actual = set(actual)
  tp = len(predicted & actual)
  tn = len(LABELS) - len(predicted | actual)
  fp = len(predicted) - tp
  fn = tn - (len(LABELS) - len(predicted))
  print(tp, fp, fn)
  p = tp / (tp + fp)
  r = tp / (tp + fn)
  b = 2
  return (1 + b**2) * p * r / (b**2*p + r)

def score(rawPredictions, expectations, threshold):
  """
  """
  predictions = get_pred(rawPredictions, threshold)
  print("score 2")
  return np.mean([F2Score(prediction, expectation) for prediction, expectation in zip(predictions, expectations)])

if __name__ == "__main__":
  with open('./train/train-idx.csv') as f_train_idx:
    train_idx = set((int(i) for i in f_train_idx.readline().split(",")))

  with open('./test/train-predict-raw.csv') as f_raw_pred:
    f_raw_pred.readline()
    raw_predictions = {}
    for l in f_raw_pred:
      img_name, values = l.split(",")
      img_idx = int(img_name.replace("train_", ""))
      values = [float(v) for v in values.split()]
      raw_predictions[img_idx] = values

  with open("./rawInput/train.csv") as f_train:
    expectations = {}
    f_train.readline()
    img_name, tags = l.split(",")
    img_idx = int(img_name.replace("train_", ""))
    if img_idx in train_idx:
      expectations[img_idx] = tags.strip().split()

  train_idx = [i for i in train_idx]

  print(score(
    [raw_predictions[i] for i in train_idx],
    [expectations[i] for i in train_idx],
    0.5
  ))
