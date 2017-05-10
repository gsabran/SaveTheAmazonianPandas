from tqdm import tqdm
import os
import shutil
from scipy.ndimage import rotate
from scipy import misc
import numpy as np
from constants import ORIGINAL_DATA_DIR, DATA_DIR, ORIGINAL_LABEL_FILE, LABEL_FILE

list_imgs = sorted(os.listdir(ORIGINAL_DATA_DIR))
try:
  shutil.rmtree(DATA_DIR)
except FileNotFoundError:
  pass
os.mkdir(DATA_DIR)

# rotate and flip images
with tqdm(total=len(list_imgs)) as pbar:
  for f in list_imgs:
    name, ext = f.split(".")
    im = misc.imread(os.path.join(ORIGINAL_DATA_DIR, f))
    for i in range(8):
      if i % 4 == 0:
        im = np.fliplr(im)
      im = rotate(im, 90)
      f2 = "{n}--{i}.{ext}".format(n=name, i=i, ext=ext)
      misc.imsave(os.path.join(DATA_DIR, f2), im)
    pbar.update(1)

# create new labels
with open(ORIGINAL_LABEL_FILE) as f, open(LABEL_FILE, "w") as f2:
  f2.write(f.readline())
  for l in f:
    filename, labels = l.split(",")
    for i in range(8):
      new_filename = "{n}--{i}".format(n=filename, i=i)
      f2.write("{name},{labels}".format(name=new_filename, labels=labels))
