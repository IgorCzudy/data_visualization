from itertools import filterfalse
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from google.colab import drive
import gzip
import keras
import glob
import re

## GET DATA
def get_train_and_test_sets(format, data_dir):
  file_pattern = f"{data_dir}/uc13-chb*-{format}-time-to-seizure*.csv.gz"
  file_paths = glob.glob(file_pattern)

  train_lines = []
  test_lines = []

  for file_path in file_paths:
      if "train" in file_path:
        data_list = train_lines
      elif "test" in file_path:
        data_list = test_lines
      else:
        # Patients < 17 go to train, rest to test
        patient_match = re.search(r"uc13-chb(\d+)-", file_path)
        patient_number = patient_match.group(1)
        data_list = train_lines if int(patient_number) < 17 else test_lines

      # df = pd.read_csv(file_path, compression="gzip", header = None)
      # data_list.extend(df.values.tolist())
      with gzip.open(file_path, 'r') as f:
        data_list += f.readlines()
        f.close()

  return train_lines, test_lines


# PREPROCESS DATA
def csv_to_tuple(line, using_cnn = False, binary_classification = False): # I'VE ADDED BINARY CLASSIFICATION
  if binary_classification:
    return csv_line_to_patient_tts_label_and_sample_binary_classification(line, using_cnn)
  else:
    return csv_line_to_patient_tts_label_and_sample_multiclass_classification(line, using_cnn)
# ------------------------------------------------------------------------------
def csv_line_to_patient_tts_label_and_sample_binary_classification(line, do_reshape = False):
  if type(line) == bytes: line = line.decode()
  parts = line.split(';')
  patient = parts[0]
  tts = float(parts[1])
  label = 0 if tts == 0 else 1
  x = np.array([float(x) for x in parts[2:]])
  if do_reshape: x = x.reshape(21, 14)
  #return (patient, tts, label, x)
  return (patient, label, x)
# ------------------------------------------------------------------------------
def csv_line_to_patient_tts_label_and_sample_multiclass_classification(line, do_reshape = False):
  if type(line) == bytes: line = line.decode()
  parts = line.split(';')
  patient = parts[0]
  tts = float(parts[1])
  label = 0
  ### BEGIN: Students can change this to check other approaches
  if tts >   2     : label = 1
  if tts >  10 * 60: label = 2
  if tts >  20 * 60: label = 3
  if tts < -10     : label = 4
  if tts < -20 * 60: label = 5
  ### END: Students can change this to check other approaches
  x = np.array([float(x) for x in parts[2:]])
  if do_reshape: x = x.reshape(21, 14)
  #return (patient, tts, label, x)
  return (patient, label, x)


def reshape_dataset(old_dataset):
  patient, label, x = old_dataset
  return (patient, label, x.reshape(21, 14))


def prepare_x(train_set, test_set, model_type, convolutional=False, ):
  X_train = np.array([t[2] for t in train_set])
  X_test  = np.array([t[2] for t in test_set])

  if convolutional:
    input_size = X_train.shape[1:] + (1,)
    X_train = X_train.reshape((X_train.shape[0], ) + input_size)
    X_test  =  X_test.reshape((X_test.shape[0], ) + input_size)
  else:
    dim = X_train.shape[1]
    i = 2
    while i < len(X_train.shape):
      dim *= X_train.shape[i]
      i += 1
    X_train = X_train.reshape(-1, dim)
    X_test = X_test.reshape(-1, dim)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if model_type == TDCNN:
      X_train = X_train.reshape(-1, 21, 14)
      X_test  =  X_test.reshape(-1, 21, 14)

  return X_train, X_test

# DATA GENERATORS
import random

class MyDataGenerator(keras.utils.Sequence):
  def __init__(self, X, y,
               shuffle = False,
               timesteps = 10,
               batch_size = 32,
               num_classes = 2,
               to_categorical = True):
    self.X = X
    self.y = y
    self.shuffle = shuffle
    self.timesteps = timesteps
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.to_categorical = to_categorical

    self.on_epoch_end()

  def on_epoch_end(self):
      self.indices = [i + self.timesteps for i in range(len(self.X) - self.timesteps)]
      if self.shuffle:
        random.shuffle(self.indices)

  def __len__(self):
    return (len(self.X) - self.timesteps) // self.batch_size

  def __getitem__(self, batch_index):
    _x_ = list()
    _y_ = list()
    for i in range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size):
      n = self.indices[i] # gets in 'n' the index of the sample in the dataset
      x = self.X[n - self.timesteps: n].copy() # gets a copy of the subsquence of several samples before the 'n'
      y = self.y[n - 1] # gets the output corresponding to 'x'
      _x_.append(x) # appends the sample 'x' to a list
      _y_.append(y) # appends the sample 'y' to a list
    #if self.to_categorical:
    #  _y_ = keras.utils.to_categorical(_y_, num_classes = self.num_classes)

    # convert to np nd-arrays both the input and the output
    return np.array(_x_), np.array(_y_)


class MyDataGeneratorToBalanceClasses(keras.utils.Sequence):
  def __init__(self, X, y,
               shuffle = False,
               batch_size = 32,
               num_classes = 2,
               to_categorical = True):
    self.X = X
    self.y = y
    self.shuffle = shuffle
    self.batch_size = (batch_size // num_classes + 1) * num_classes
    self.num_classes = num_classes
    self.to_categorical = to_categorical

    # One index per target class is created in order to make it easy balancing
    # the batches with the same, approximately, number of samples of each
    # target class per batch
    self.indices = list()
    for c in range(num_classes):
      self.indices.append([i for i in filter(lambda a: np.argmax(y[a]) == c, range(len(y)))])

    self.on_epoch_end()

  def on_epoch_end(self):
      self.current_index = list()
      for c in range(len(self.indices)):
        self.current_index.append(0)
        if self.shuffle:
          random.shuffle(self.indices[c])

  def __len__(self):
    return 1 + max([len(l) for l in self.indices]) // self.batch_size

  def __getitem__(self, batch_index):
    _x_ = list()
    _y_ = list()
    while len(_x_) < self.batch_size:
      for c in range(len(self.indices)):
        l = len(self.indices[c])
        if l > 0: # 2022-12-30 checking to prevent computing the module when no samples of a given class exist in the training subset
          n = self.current_index[c] # get the index for the next sample in target class 'c'
          self.current_index[c] = (n + 1) % l # update the index of the next sample
          n = self.indices[c][n] # get in 'n' now the index of the sample in the dataset
          x = self.X[n].copy() # get the input in 'x'
          y = self.y[n] # get the output label (or value) in 'y'
          _x_.append(x) # append the sample 'x' in a list
          _y_.append(y) # append the sample 'y' in a list
    #if self.to_categorical:
    #  _y_ = keras.utils.to_categorical(_y_, num_classes = self.num_classes)

    # convert to Numpy nd-arrays both input and output
    return np.array(_x_), np.array(_y_)

# EXPERIMENT CONFIGURATION
FEEDFORWARD = 1
CONVOLUTIONAL = 2
RECURRENT = 3
TDCNN = 4

class LabConfiguration:
  def __init__(self, model_type = FEEDFORWARD,
               dnn = None,
               to_categorical = True,
               timesteps = 1,
               epochs = 1):
    self.model_type = model_type
    self.dnn = dnn
    self.to_categorical = to_categorical
    self.timesteps = timesteps
    self.epochs = epochs
    assert self.dnn is not None


# GET RESULTS

def get_predictions(model, conf, y_train, y_test, X_train, X_test, train_data_generator=None, test_data_generator=None):
    y_train_true = None
    y_test_true = None
    y_train_pred = None
    y_test_pred = None

    if conf.model_type == RECURRENT or conf.model_type == TDCNN:
        train_data_generator.shuffle = False
        train_data_generator.on_epoch_end()
        y_train_pred = model.predict(train_data_generator)
        y_test_pred = model.predict(test_data_generator)

        y_train_true = y_train.argmax(axis = 1) if conf.to_categorical else y_train
        y_test_true = y_test.argmax(axis = 1) if conf.to_categorical else y_test
        y_train_true = y_train_true[-len(y_train_pred):]
        y_test_true = y_test_true[-len(y_test_pred):]

    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_true = y_train.argmax(axis = 1) if conf.to_categorical else y_train
        y_test_true = y_test.argmax(axis = 1) if conf.to_categorical else y_test


    y_train_pred = y_train_pred.argmax(axis = 1)
    y_test_pred = y_test_pred.argmax(axis = 1)

    return y_train_true, y_train_pred, y_test_true, y_test_pred
