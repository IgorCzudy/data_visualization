from matplotlib import pyplot
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
import matplotlib.pyplot as plt

def show_results(y_train_true, y_train_pred, y_test_true, y_test_pred, labels, history, model_identifier, parent_dir, model_name):
  his_results(y_train_true, y_train_pred, labels)
  his_results(y_test_true, y_test_pred, labels)
  plot_training_loss_and_metric(history, model_identifier, parent_dir, model_name)


def his_results(y_true, y_pred, labels):
  _cm_ = confusion_matrix(y_true, y_pred)
  for i in range(_cm_.shape[0]):
      for j in range(_cm_.shape[1]):
          print(' %10d' % _cm_[i, j], end = ' ')
      print('\n')

  print(classification_report(y_true, y_pred))
  #
  fig, axes = pyplot.subplots(nrows = 1, ncols = 2, figsize = (16, 7))
  #fig.suptitle(title)
  #
  ConfusionMatrixDisplay.from_predictions(y_true = y_true, y_pred = y_pred,
                                          display_labels = labels,
                                          normalize = 'true', ax = axes[0],
                                          cmap = pyplot.cm.Blues) #, colorbar = False)
  #
  ConfusionMatrixDisplay.from_predictions(y_true = y_true, y_pred = y_pred,
                                          display_labels = labels,
                                          normalize = 'pred', ax = axes[1],
                                          cmap = pyplot.cm.Oranges) #, colorbar = False)
  #
  pyplot.tight_layout()
  pyplot.show()
  del fig

### PLOTS
def plot_training_loss_and_metric(history, model_identifier, parent_dir, model_name):
  if history is None:
    print("Cannot print training loss and metrics when reading from checkpoint")
  else:
    # plotter = tfdocs.plots.HistoryPlotter(metric = 'sparse_categorical_accuracy', smoothing_std=10)
    # plotter.plot({f"{conf.dnn.__name__}": history})
    metric = "sparse_categorical_accuracy" if "sparse_categorical_accuracy" in history.history.keys() else "categorical_accuracy"
    metric_name = " ".join(metric.split("_"))

    plt.figure(figsize=(10,6))
    plt.plot(history.epoch, history.history[metric], color="red", label=metric_name)
    plt.plot(history.epoch, history.history[f'val_{metric}'], color="red", linestyle="dashed", label=f'Val {metric_name}')
    plt.plot(history.epoch, history.history['loss'], color="blue", label='Loss')
    plt.plot(history.epoch, history.history['val_loss'], color="blue", linestyle='dashed', label='Val loss')

    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name} / loss')

    plt.tight_layout()
    plt.legend(loc='upper left')

    plt.title(f'Loss and {metric_name} for {model_name}')
    plt.savefig(f"{parent_dir}/TEAA/plots/training_loss_metric_{model_identifier}.png")
    plt.show()


### RAW
def get_raw_metrics(y_true, y_pred):
  metrics = dict()
  metrics["macro_accuracy"] = accuracy_score(y_true, y_pred)
  metrics["macro_precision"] = precision_score(y_true, y_pred, average='macro')
  metrics["macro_recall"] = recall_score(y_true, y_pred, average='macro')
  metrics["macro_f1"] = f1_score(y_true, y_pred, average='macro')
  return metrics

def save_raw_results(y_train_true, y_train_pred, y_test_true, y_test_pred, model_identifier, parent_dir):
  train_metrics = get_raw_metrics(y_train_true, y_train_pred)
  test_metrics = get_raw_metrics(y_test_true, y_test_pred)   
  with open(f"{parent_dir}/TEAA/teaa4_models_results.csv", "a") as file:
    file.write(f"{';'.join(model_identifier.split('_'))};")
    file.write(";".join(map(lambda x: str(x), train_metrics.values())))
    file.write(";")
    file.write(";".join(map(lambda x: str(x), test_metrics.values())))
    file.write("\n")

def open_raw_results(parent_dir):
  metrics = ["accuracy", "precision", "recall", "f1"]
  colnames = ["pca", "model", "epochs", "e", "bin_multi"] + list(map(lambda x: "train_" + x, metrics)) + list(map(lambda x: "test_" + x, metrics))
  return pd.read_csv(f"{parent_dir}/TEAA/teaa4_models_results.csv", sep = ';', names = colnames).drop(columns="e")


