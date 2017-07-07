from .dataset import Dataset
from constants import LABELS, WEATHER_IDX

class WeatherDataset(Dataset):
	"""
	A dataset with only the weathers labels
	"""
	labels = [l for i, l in enumerate(LABELS) if i in WEATHER_IDX]


class FilteredDataset(Dataset):
  """
  A dataset with only part of the data
  """

  def __init__(self, list_files, required_tag, validation_ratio=0, sessionId=None, training_files=None, validation_files=None):
    self.required_tag = required_tag
    super(FilteredDataset, self).__init__(
      list_files,
      validation_ratio=validation_ratio,
      sessionId=sessionId,
      training_files=training_files,
      validation_files=validation_files
    )

  def select_files(self, list_files):
    """
    Select files that can be part of the dataset based on a criteria
    """
    tag_idx = LABELS.index(self.required_tag)
    return [f for f in list_files if self.outputs[f][tag_idx] == 1]
