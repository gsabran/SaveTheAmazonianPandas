from tqdm import tqdm
import os
import shutil
from constants import ORIGINAL_DATA_DIR, DATA_DIR, ORIGINAL_LABEL_FILE, LABEL_FILE

from subprocess import call

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
    angle = 0
    flip = False
    for i in range(8):
      if i == 4:
        flip = True
        angle = 0
      f2 = "{n}--{i}.{ext}".format(n=name, i=i, ext=ext)
      if flip:
        call(["convert", "-rotate", str(angle), "-flip", os.path.join(ORIGINAL_DATA_DIR, f), os.path.join(DATA_DIR, f2)])
      else:
        call(["convert", "-rotate", str(angle), os.path.join(ORIGINAL_DATA_DIR, f), os.path.join(DATA_DIR, f2)])
      angle += 90
    pbar.update(1)

# create new labels
with open(ORIGINAL_LABEL_FILE) as f, open(LABEL_FILE, "w") as f2:
  f2.write(f.readline())
  for l in f:
    filename, labels = l.split(",")
    for i in range(8):
      new_filename = "{n}--{i}".format(n=filename, i=i)
      f2.write("{name},{labels}".format(name=new_filename, labels=labels))
