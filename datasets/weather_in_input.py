import numpy as np

from .dataset import Dataset
from constants import LABELS, WEATHER_IDX, TRAIN_DATA_DIR
from utils import get_resized_image


class WeatherInInputDataset(Dataset):
  """
  A dataset with only the weathers labels in input
  """
  
  input_length = 2

  def getXY(self, file_names):
    Y = None
    X = None
    X2 = None
    for f in file_names:
      file = f.split('/')[-1].split('.')[0]
      img, weather_tags = self.get_input(f, TRAIN_DATA_DIR)
      y = self.outputs[file]
      if X is None:
        X = np.zeros((len(file_names),) + img.shape)
        X2 = np.zeros((len(file_names),) + weather_tags.shape)
        Y = np.zeros((len(file_names),) + y.shape)
      X.append(img)
      X2.append(weather_tags)
      Y.append(y)
    return [[np.array(X), np.array(X2)], np.array(Y)]

  def get_input(self, image_name, data_dir):
    """
    Return the input corresponding to one image file
    """
    img = get_resized_image(
      image_name,
      data_dir,
      image_data_fmt,
      input_shape,
      self._image_normalization
    )
    return img, [self.outputs[image_name][i] for i in WEATHER_IDX]
