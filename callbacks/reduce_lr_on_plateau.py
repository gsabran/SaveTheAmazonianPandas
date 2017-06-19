from keras.callbacks import Callback
from keras import backend as K
import numpy as np

class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a "patience" number
    of epochs, the learning rate is reduced.
    # Example
        ```python
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                      patience=5, min_lr_factor=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
    # Arguments
        model: the instance of the model (not the Keras model)
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr_factor: lower bound on the learning factor rate.
        checkpoint_path: a path where best checkpoint is saved
    """

    def __init__(self, model, monitor="val_loss", factor=0.1, patience=10,
                 verbose=0, mode="auto", epsilon=1e-4, cooldown=0, min_lr_factor=0,
                 checkpoint_path=None):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError("ReduceLROnPlateau "
                             "does not support a factor >= 1.0.")
        self.factor = factor
        self.min_lr_factor = min_lr_factor
        self.min_lr = 0
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.amazonian_model = model
        self.checkpoint_path = checkpoint_path
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ["auto", "min", "max"]:
            print("Learning Rate Plateau Reducing mode {mode} is unknown, fallback to auto mode.".format(mode=self.mode))
            raise RuntimeWarning
        if (self.mode == "min" or
           (self.mode == "auto" and "acc" not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def on_train_begin(self, logs=None):
        self.min_lr = self.min_lr_factor * float(K.get_value(self.model.optimizer.lr))
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            print("Learning Rate Plateau Reducing requires {mode} available!".format(mode=self.monitor))
            print(logs)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best): 
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr + self.lr_epsilon:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        if self.checkpoint_path is not None:
                            print("loading past weights")
                            self.model.load_weights(self.checkpoint_path)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        print("\nEpoch {epoch}: reducing learning rate to {new_lr}.".format(epoch=epoch, new_lr=new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0
