import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score)

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    #specyficzność osobno dla każdej klasy, potem średnia
    specs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        if (tn + fp) > 0:
            specs.append(tn / (tn + fp))
        else:
            specs.append(0.0)
    spec = float(np.mean(specs))

    return {"accuracy": acc,"precision_macro": prec,"recall_macro": rec,"specificity_macro": spec,"f1_macro": f1,"balanced_accuracy": bal_acc,}

#Macierze pomyłek
def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Macierz pomyłek"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)

    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
