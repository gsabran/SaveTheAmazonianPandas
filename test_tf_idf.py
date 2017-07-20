from constants import TRAIN_DATA_DIR
from datasets.dataset import Dataset
import os, random
from tqdm import tqdm

list_imgs = [f.split(".")[0] for f in sorted(os.listdir(TRAIN_DATA_DIR))]

data = Dataset(list_imgs, 0.0, "")

gen = data.batch_generator(1, (3, 256, 256))

start = True
for i in tqdm(range(int(1e4))):
	features, labels = next(gen)
	if start:
		all_labels = labels
		start = False
	else:
		all_labels += labels

print(all_labels)


# This is code to generate labels without balancing for comparison
"""
gen = data.batch_generator(1, (3, 256, 256), balance=False)

start = True
for i in tqdm(range(int(1e4))):
	features, labels = next(gen)
	if start:
		all_labels = labels
		start = False
	else:
		all_labels += labels
print(all_labels)"""