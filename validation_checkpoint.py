from keras.callbacks import Callback

class ValidationCheckpoint(Callback):
	"""
	Callback tomeasure performance on validation set
	- Parameter scoring: a function that takes (model, input, output) and returns a score
	- Parameter patience: the number of epoch without improvement that can be tolerated
	"""

	def __init__(self, scoring, validation_input, validation_output, patience=1):
		super(ValidationCheckpoint, self).__init__()
		self.scoring = scoring
		self.validation_input = validation_input
		self.validation_output = validation_output
		self.best_score = 0
		self.best_epoch = -1
		self.patience = patience
		self.remaining_patience = patience

	def on_epoch_end(self, epoch, logs=None):
		print("Scoring validation_inputdation set...".format(epoch=epoch))
		score = self.scoring(self.model, self.validation_input, self.validation_output)
		print("Validation score is {score} (previous score was {previous_score})".format(score=score, previous_score=self.best_score))
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
