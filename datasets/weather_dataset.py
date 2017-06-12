from .dataset import Dataset
from constants import LABELS, WEATHER_IDX

class WeatherDataset(Dataset):
	"""
	A dataset with only the weathers labels
	"""
	labels = [l for i, l in enumerate(LABELS) if i in WEATHER_IDX]
