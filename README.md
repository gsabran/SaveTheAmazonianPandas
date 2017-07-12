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
Create or load a model and the training data, and fit the model. The model is saved after each epoch under `train/checkpoint.hdf5` and the final model is saved under `train/model.h5`. The data used for training is written under `train/training-files.csv` and remaining labeled inputs can be used for validation.

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

## Attemps

| f2 test | f2 validation | model | epochs | generationg data | other |
|---|---|---|---|---|---|
| 0.90862 | 0.9063 | ekami | 50 | no | early-stopping, auto LR decrease on accuracy |
| 0.9105 | 0.8963 | ekami | 50 | no | early-stopping, auto LR decrease on f2 validation (might not work) |
| 0.909 | 0.9075 | ekami | 50 | no | early-stopping, auto LR decrease on f2 validation |
| 0.909 | 0.9075 | ekami | 50 | no | early-stopping, auto LR decrease on f2 validation |
| 0.9205 | 0.9190 | ekami | 100 | no | early-stopping, auto LR decrease on f2 validation |
| 0.9205 | 0.9192 | ekami | 100 | no | early-stopping, auto LR decrease on f2 validation gaussian white noise |
| 0.91312 | 0.9156 | ekami | 100 | no | same, full size images |
| _ | 0.94 | ekami | 100 | no | truth used in final weather labels |
| _ | 0.946 | gui | 100 | no | true weather used in dense layer |
| _ | 0.82 | densenet121 | 100 | no | early stopping - unsure if reliable |

