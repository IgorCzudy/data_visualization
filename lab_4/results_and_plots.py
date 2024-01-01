from matplotlib import pyplot

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
import matplotlib.pyplot as plt

def show_results(y_true, y_pred, labels):
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

def plot_training_loss_and_metric(history, model_identifier):
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

  plt.title(f'Loss and {metric_name} for {conf.dnn.__name__}')
  plt.savefig(f"{parent_dir}/TEAA/plots/training_loss_metric_{model_identifier}.png")
  plt.show()