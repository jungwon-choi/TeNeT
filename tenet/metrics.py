import numpy as np
from sklearn import metrics


def evaluate(true, pred, prob, num_classes):
    """Classification metrics (AUROC, accuracy, ...) or regression metrics (R^2, MSE, MAE)."""
    if num_classes == 1:
        return {
            'r2': metrics.r2_score(true, pred),
            'mse': metrics.mean_squared_error(true, pred),
            'mae': metrics.mean_absolute_error(true, pred),
            'explained_var': metrics.explained_variance_score(true, pred),
        }
    average = 'binary' if num_classes == 2 else 'micro'
    if num_classes == 2:
        roc_auc = metrics.roc_auc_score(true, prob[:, 1])
    else:
        roc_auc = metrics.roc_auc_score(true, prob, average='macro', multi_class='ovr')
    return {
        'roc_auc': roc_auc,
        'accuracy': metrics.accuracy_score(true, pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(true, pred),
        'precision': metrics.precision_score(true, pred, average=average),
        'recall': metrics.recall_score(true, pred, average=average),
        'f1': metrics.f1_score(true, pred, average='macro'),
    }
