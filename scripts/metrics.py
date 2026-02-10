import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score,balanced_accuracy_score)

animals = ["cheetah", "elephant", "giraffe", "lion", "rhino", "zebra"]

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    #specificity licze "one-vs-rest" dla każdej klasy i uśredniam
    specificity_list = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])

        if (tn + fp) > 0:
            specificity_list.append(tn / (tn + fp))
        else:
            specificity_list.append(0)

    specificity = float(np.mean(specificity_list))
    return accuracy, precision, recall, specificity, f1, balanced_acc

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)

    #etykiety --> najpierw argument labels, a jak nie ma to animals
    if labels is None:
        labels_to_use = animals
    else:
        labels_to_use = labels

    #zabezpieczeni --> dopasuj długość etykiet do rozmiaru cm
    n = cm.shape[0]
    if labels_to_use is not None:
        labels_to_use = list(labels_to_use)
        if len(labels_to_use) != n:
            labels_to_use = labels_to_use[:n] + [""] * (n - len(labels_to_use))

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_to_use)
    disp.plot(cmap="Blues", values_format="d")

    plt.title("Macierz pomyłek")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
