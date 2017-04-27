# Installation
- download and extract the raw data files and put them in `rawInput`
```bash
python ./data-processing/process.py
```

# Settings
- python 3
- tab indentation

# Usage

## Training
Create or load a model and the training data, and fit the model. The model is saved after each epoch under `train/checkpoint.hdf5` and the final model is saved under `train/model.h5`. The data used for training is written under `train/train-idx.csv` and remaining labeled inputs can be used for validation.

```bash
python3 model.py --epoch 10 --batch-size 16 --test-ratio 0.2 --gpu 8
```

In case of unexpected interruption, the partially trained model can be reloaded and finish fitting:
```bash
python3 model.py --epoch 4 --batch-size 16 --gpu 8 --model train/checkpoint.hdf5
```

## Predict
Prediction can be done on the test set or the train set. They output two files: one with the probabilities for each label, and one with the final labels
```bash
python3 predict.py --model train/model.h5 --data test
```
```bash
python3 predict.py --model train/model.h5 --data train
```