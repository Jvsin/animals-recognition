import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,f1_score, balanced_accuracy_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = float(np.trace(cm) / np.sum(cm)) #liczby musza byc float
    precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    specificity_list = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(spec_i)
    specificity = float(np.mean(specificity_list))

    return accuracy, precision, recall, specificity, f1, balanced_acc

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
