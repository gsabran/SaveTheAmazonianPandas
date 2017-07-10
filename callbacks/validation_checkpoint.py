from shutil import copyfile
from keras.callbacks import Callback

class ValidationCheckpoint(Callback):
	"""
	Callback tomeasure performance on validation set
	- Parameter scoring: a function that takes (model, input, output) and returns a score
	- Parameter patience: the number of epoch without improvement that can be tolerated
	- Parameter checkpoint_path: path to where the checkpoint of the best model should be saved
	- Parameter sessionId: an id to easily identify the running session
	"""

	def __init__(self, scoring, training_input, training_output, validation_input, validation_output, checkpoint_path, sessionId, patience=1):
		super(ValidationCheckpoint, self).__init__()
		self.scoring = scoring
		self.training_input = training_input
		self.training_output = training_output
		self.validation_input = validation_input
		self.validation_output = validation_output
		self.best_score = 0
		self.best_epoch = -1
		self.patience = patience
		self.remaining_patience = patience
		self.checkpoint_path = checkpoint_path
		self.sessionId = sessionId

	def on_epoch_end(self, epoch, logs=None):
		logs = logs if logs is not None else {}
		print("\nScoring validation dataset...".format(epoch=epoch))
		train_score = self.scoring(self.model, self.training_input, self.training_output)
		logs["f2_train_score"] = train_score

		if len(self.validation_input) == 0:
			print("No data provided to validate model")
			score = 0.0
		else:
			score = self.scoring(self.model, self.validation_input, self.validation_output)
		print("\nValidation score is {score} (previous score was {previous_score})".format(score=score, previous_score=self.best_score))
		print("Traning score is {score}".format(score=train_score))
		logs["f2_val_score"] = score
		if score <= self.best_score:
			self.remaining_patience -= 1
			if self.remaining_patience == 0:
				self.model.stop_training = True
				print("Stopping training due to lack of improvement. Best epoch is #{best}".format(best=self.best_epoch + 1)) # number starts at 1
				return
		else:
			self.best_score = score
			self.remaining_patience = self.patience
			self.best_epoch = epoch
			self.model.save(self.checkpoint_path, overwrite=True)
			copyfile(self.checkpoint_path, "train/archive/{id}-best-checkpoint.hdf5".format(id=self.sessionId))
