from matplotlib import pyplot

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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