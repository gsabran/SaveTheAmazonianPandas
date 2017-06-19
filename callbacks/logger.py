from keras.callbacks import Callback
from utils import remove

class Logger(Callback):
	"""
	Callback tomeasure performance on validation set
	- Parameter scoring: a function that takes (model, input, output) and returns a score
	- Parameter patience: the number of epoch without improvement that can be tolerated
	- Parameter checkpoint_path: path to where the checkpoint of the best model should be saved
	"""

	def __init__(self, destination="./train/training.logs"):
		super(Logger, self).__init__()
		remove(destination)
		self.destination = destination
		self.logs_keys = None

	def on_epoch_end(self, epoch, logs=None):
		if self.logs_keys is None:
			self.logs_keys = [k for k in logs]
			self.csv_file.write("epoch,{labels}\n".format(labels=",".join(self.logs_keys)))

		data = [logs[k] for k in self.logs_keys]
		data.insert(0, epoch)
		self.csv_file.write("{data}\n".format(data=",".join([str(x) for x in data])))
		print("Epoch {epoch} logs: {logs}".format(epoch=epoch, logs=logs))
		print("\n")

	def on_train_begin(self, logs=None):
		self.csv_file = open(self.destination, 'w')

	def on_train_end(self, logs=None):
		self.csv_file.write("Training over\n")
		self.csv_file.close()